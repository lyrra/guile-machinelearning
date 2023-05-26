(define-module (guile-machinelearning net)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 match)
  #:use-module (guile-machinelearning common)
  #:use-module (guile-machinelearning common-lisp)
  #:use-module (guile-machinelearning bio)
  #:use-module (guile-machinelearning mat)
  #:use-module (guile-gpu gpu)
  #:use-module (guile-gpu sigmoid)
  #:use-module (ffi cblas) ; needed for guile-ffi-cblas wrapper saxpy!
  #:export (<netr>
            make-net
            netr?
            netr-info set-netr-info!
            netr-numin set-netr-numin!
            netr-numout set-netr-numout!
            netr-numhid set-netr-numhid!
            netr-in   set-netr-in!
            netr-out  set-netr-out!
            netr-hid  set-netr-hid!
            netr-act  set-netr-act!
            netr-grad set-netr-grad!
            netr-arrs set-netr-arrs!
            netr-wdelta set-netr-wdelta!
            net-grad-clone
            net-grad-clear
            net-wdelta-clear
            net-vyo
            net-vxi
            update-eligibility-traces
            net-weights-scale
            net-make-wdelta
            net-accu-wdelta
            update-weights
            net-add-wdelta
            net-run
            net-set-input
            net-copy
            net-merge!
            net-transfer
            file-write-net
            file-load-net
            ; not public, but used by tests
            net--read-arrays
            net--write-arrays))

(define-record-type <netr>
  (make-netr)
  netr?
  (info netr-info set-netr-info!)
  (numin netr-numin set-netr-numin!)
  (numout netr-numout set-netr-numout!)
  (numhid netr-numhid set-netr-numhid!)
  (in   netr-in   set-netr-in!)
  (out  netr-out  set-netr-out!)
  (hid  netr-hid  set-netr-hid!)
  (act  netr-act  set-netr-act!)
  (grad netr-grad set-netr-grad!)
  ; old
  (arrs netr-arrs set-netr-arrs!)
  (wdelta netr-wdelta set-netr-wdelta!))

(define (net--read-arrays p)
  (let* ((arrlen (bio-read-uint32 p))
         (net (make-array #f arrlen)))
    (do ((n 0 (1+ n)))
        ((>= n arrlen))
      (let ((arr (bio-read-array/matrix p)))
        (array-set! net arr n)))
    net))

(define (net--write-arrays p arrs)
  (bio-write-uint32 p (array-length arrs))
  (array-for-each (lambda (gpu-arr)
                    (gpu-refresh-host gpu-arr)
                    (bio-write-array/matrix p (gpu-array gpu-arr)))
                  arrs))

(define (file-load-net file)
  (call-with-input-file file
    (lambda (p)
      (let ((ver (bio-read-uint32 p))) ; version
        (cond
         ((= ver 1)
          (let* ((episode (bio-read-uint32 p))
                 (net (net--read-arrays p))
                 (net2 (net-make-from net #f)))
            (set-netr-info! net2 (list 'episode episode))
            net2))
         (else
          (let ((episode (bio-read-uint32 p))
                (numin  (bio-read-uint32 p))
                (numout (bio-read-uint32 p))
                (numhid (bio-read-uint32 p)))
            (let* ((arrs (net--read-arrays p))
                   (net2 (make-net #:init #f #:in numin #:out numout #:hid numhid))
                   (arrs2 (netr-arrs net2)))
              (set-netr-info! net2 (list 'episode episode))
              (do ((i 0 (+ i 1))) ((>= i (array-length arrs)))
                (gpu-array-copy (array-ref arrs2 i) (array-ref arrs i)))
              net2))))))
    #:guess-encoding #f
    #:encoding #f
    #:binary #t))

(define (file-write-net file episode net)
  (call-with-output-file file
    (lambda (p)
      (bio-write-uint32 p 2) ; version
      (bio-write-uint32 p episode)
      (bio-write-uint32 p (netr-numin  net))
      (bio-write-uint32 p (netr-numout net))
      (bio-write-uint32 p (netr-numhid net))
      (net--write-arrays p (netr-arrs net)))
    #:encoding #f #:binary #t))

(define (net-make-wdelta net)
  (let ((wdeltaarr (make-array #f 2))
        (in  (netr-numin net))
        (out (netr-numout net))
        (hid (netr-numhid net)))
    (array-set! wdeltaarr (gpu-make-matrix hid in)  0) ; mhw
    (array-set! wdeltaarr (gpu-make-matrix out hid) 1) ; myw
    (set-netr-wdelta! net wdeltaarr)))

(define* (make-net #:key (init #t) in out hid (wdelta #f))
  (let ((netr (make-netr))
        (net (make-array #f 7)))
    (array-set! net (gpu-make-matrix hid in)  0) ; mhw
    (array-set! net (gpu-make-vector hid)     1) ; vhz
    (array-set! net (gpu-make-vector hid)     2) ; vho
    (array-set! net (gpu-make-matrix out hid) 3) ; myw
    (array-set! net (gpu-make-vector out)     4) ; vyz
    (array-set! net (gpu-make-vector out)     5) ; vyo
    (array-set! net (gpu-make-vector in)      6) ; vxi
    ; setup weight-change cache
    (when wdelta
      (let ((wdeltaarr (make-array #f 2)))
        (array-set! wdeltaarr (gpu-make-matrix hid in)  0) ; mhw
        (array-set! wdeltaarr (gpu-make-matrix out hid) 1) ; myw
        (set-netr-wdelta! netr wdeltaarr)))
    ;
    (set-netr-arrs!   netr net)
    (set-netr-numin!  netr  in)
    (set-netr-numout! netr out)
    (set-netr-numhid! netr hid)
    (set-netr-grad! netr
                    ; eligibility traces, 0-1 is index in output-layer
                    (list (do ((i 0 (1+ i))
                               (lst '()))
                              ((= i out) lst)
                            (set! lst (cons (gpu-make-matrix hid in) lst)))
                          (do ((i 0 (1+ i))
                               (lst '()))
                              ((= i out) lst)
                            (set! lst (cons (gpu-make-vector hid) lst)))))
    (if init
      (array-for-each (lambda (arr)
       (gpu-array-apply arr (lambda (x) (* 0.01 (- (random-uniform) .5)))))
                 net))
    netr))

(define (net-free netr)
  (let ((arrs (netr-arrs netr)))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (gpu-free-array (array-ref arrs i)))))

(define (net-serialize netr)
  (let ((arrs (netr-arrs netr))
        (arrs2 (make-array #f 7)))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (gpu-refresh-host (array-ref arrs i))
      (array-set! arrs2 (gpu-array (array-ref arrs i)) i))
    arrs2))

(define* (net-make-from arrs #:optional (init #t))
  (let* ((in  (array-length (array-ref arrs 6)))
         (out (array-length (array-ref arrs 5)))
         (hid (array-length (array-ref arrs 1)))
         (net2 (make-net #:init init #:in in #:out out #:hid hid))
         (arrs2 (netr-arrs net2)))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (gpu-array-copy (array-ref arrs2 i) (array-ref arrs i)))
    net2))

(define (net-grad-clone net)
  (map-in-order (lambda (lst) (map-in-order gpu-array-clone lst))
                (netr-grad net)))

(define (net-grad-clear eligs)
  (loop-for lst in eligs do
    (loop-for arr in lst do
      (gpu-array-apply arr (lambda (x) 0.)))))

(define (net-wdelta-clear net)
  (array-for-each (lambda (wdelta)
                    (gpu-array-apply wdelta (lambda (x) 0.)))
                  (netr-wdelta net)))

; get array's as refreshed host-arrays
(define (net-vyo netr)
  (let* ((arrs (netr-arrs netr))
         (rv (array-ref arrs 5)))
    (gpu-refresh-host rv)
    (gpu-array rv)))
(define (net-vxi netr)
  (let* ((arrs (netr-arrs netr))
         (rv (array-ref arrs 6)))
    (gpu-refresh-host rv)
    (gpu-array rv)))

(define (net-set-input netr input)
  (let ((arrs (netr-arrs netr)))
    (gpu-array-copy (array-ref arrs 6) input)))

(define (net-copy src)
  (let* ((arrs (netr-arrs src))
         (in  (gpu-rows (array-ref arrs 6)))
         (out (gpu-rows (array-ref arrs 5)))
         (hid (gpu-rows (array-ref arrs 1)))
         (dst (make-net #:init #f #:in in #:out out #:hid hid
                        #:wdelta (if (netr-wdelta src) #t #f))))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (gpu-array-map! (array-ref (netr-arrs dst) i)
                      (lambda (x) x)
                      (array-ref arrs i)))
    dst))

(define (net-transfer dst src)
  (let ((darrs (netr-arrs dst))
        (sarrs (netr-arrs src)))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (let ((srv (gpu-array (array-ref sarrs i)))
            (drv (gpu-array (array-ref darrs i))))
        (array-scopy! srv drv)))))

(define (net-merge! dst src1 src2 take-ratio)
  (let ((darrs (netr-arrs dst))
        (s1arrs (netr-arrs src1))
        (s2arrs (netr-arrs src2))
        (keep-ration (- 1 take-ratio)))
    (array-for-each (lambda (drv vr1 vr2)
                      (array-map! (gpu-array drv)
                                  (lambda (a b)
                                    (+ (* b take-ratio)
                                       (* a keep-ration)))
                                  (gpu-array vr1) (gpu-array vr2)))
                    darrs s1arrs s2arrs)))

(define (net-run net input)
  (match (netr-arrs net)
    (#(mhw vhz vho myw vyz vyo vxi)
     (gpu-array-copy vxi input)
     (gpu-sgemv! 1. mhw #f vxi 0. vhz)
     (gpu-array-sigmoid vhz vho)
     (gpu-sgemv! 1. myw #f vho 0. vyz)
     (gpu-array-sigmoid vyz vyo)
     #f)))

(define (net-weights-scale net proc alpha)
  (let* ((arrs (netr-arrs net))
         (wdelta (netr-wdelta net))
         (mhw (array-ref arrs 0))
         (myw (array-ref arrs 3)))
    (cond
     (wdelta
      (gpu-array-map2! mhw (lambda (w e) (proc 0 alpha w e)) mhw (array-ref wdelta 0))
      (gpu-array-map2! myw (lambda (w e) (proc 1 alpha w e)) myw (array-ref wdelta 1)))
     (else
      (gpu-array-apply mhw (lambda (w) (proc 0 alpha w)))
      (gpu-array-apply myw (lambda (w) (proc 1 alpha w)))))))

(define (net-accu-wdelta net alpha tderr grads)
  (let* ((arrs (netr-arrs net))
         (len (array-length arrs))
         (eliglays (reverse grads))
         (numout (array-length tderr))
         (numweil (array-length (netr-wdelta net))))
    (do ((l (- len 1 3) (- l 3))
         (el 0 (1+ el)))
        ((< l 0))
      ;(format #t "el: ~s~%" (- numweil el 1))
      (let* (;(mw (array-ref arrs l)) ; weight-layer
             (mw (array-ref (netr-wdelta net) (- numweil el 1)))
             (eligs (list-ref eliglays el))
             (ei (length eligs))
             (oi (gpu-rows mw)))
        ;(format #t "mw-dim: ~s <=> ~s~%" (gpu-array-dimensions mw)
        ;                                 (gpu-array-dimensions mw2))
        (do ((e 0 (+ e 1))) ((= e ei)) ; for each eligibility mirror
          (do ((i 0 (+ i 1))) ((= i (gpu-rows mw))) ; for each neuron in this layer
            (let ((tde (* alpha (array-ref tderr (if (= 0 el) i e)))))
              (if (or (> el 0) (= i e))
                  (gpu-saxpy! tde (list-ref eligs e) mw
                              (if (= 0 el) #f i) ; elig at output-layer is a vector
                              i)))))))))

(define (net-add-wdelta net)
  (let* ((arrs (netr-arrs net))
         (len (array-length arrs))
         (numweil (array-length (netr-wdelta net))))
    (do ((l (- len 1 3) (- l 3))
         (el 0 (1+ el)))
        ((< l 0))
      (let* ((mw (array-ref arrs l)) ; weight-layer
             (wd (array-ref (netr-wdelta net) (- numweil el 1))))
        (gpu-refresh-host mw)
        (gpu-refresh-host wd)
        (let ((mwarr (gpu-array mw))
              (wdarr (gpu-array wd)))
          (array-map! mwarr (lambda (a b) (+ a b)) mwarr wdarr))
        (gpu-dirty-set! mw 1)))))

; gradient-descent, return weight update in grads
; historically this was backprop, therefore we update the layers in reverse
(define (update-weights net alpha tderr grads)
  (let* ((arrs (netr-arrs net))
         (len (array-length arrs))
         (eliglays (reverse grads))
         (numout (array-length tderr)))
    (do ((l (- len 1 3) (- l 3))
         (el 0 (1+ el)))
        ((< l 0))
      (let* ((mw (array-ref arrs l)) ; weight-layer
             (eligs (list-ref eliglays el))
             (ei (length eligs))
             (oi (gpu-rows mw)))
        (do ((e 0 (+ e 1))) ((= e ei)) ; for each eligibility mirror
          (do ((i 0 (+ i 1))) ((= i (gpu-rows mw))) ; for each neuron in this layer
            (let ((tde (* alpha (array-ref tderr (if (= 0 el) i e)))))
              (if (or (> el 0) (= i e))
                  (gpu-saxpy! tde (list-ref eligs e) mw
                              (if (= 0 el) #f i) ; elig at output-layer is a vector
                              i)))))))))

; discount eligibility traces
; elig  <- gamma*lambda * elig + Grad_theta(V(s))
; z <- y*L* + Grad[V(s,w)]
(define (update-eligibility-traces net eligs gamlam)
  (loop-for lst in eligs do
    (loop-for gar in lst do
      ;(gpu-sscal! gamlam gar) ; gives discrepancies
      (gpu-array-apply gar (lambda (x) (* gamlam x)))))
  (let* ((arrs (netr-arrs net))
         (len (array-length arrs))
         (vxi (array-ref arrs (1- len)))
         (numhid (netr-numhid net))
         (numout (netr-numout net))
         (nextg #f) ; next-to-be-calculated gradient, towards input-layer
         (eliglays (reverse eligs)))
      ; go through the networks layers starting from the output layer
      ; the very last layer holds a copy of the input, so skip that
      ; the gradients are propagated backwards through nextg
      (do ((l (- len 1 3) (- l 3))  ; 1= skip input layer
           (el 0 (1+ el))) ; elig-trace layer
          ((< l 0))
        (let* ((mx (array-ref arrs (if (< (- l 1) 0)
                                     (1- len) ; if at network-input layer, use input-layer as input
                                     (1- l)))) ; else use the above layer as input
               (mw (array-ref arrs l)) ; weight-layer
               (mz (array-ref arrs (+ l 1))) ; linear-layer
               (ma (array-ref arrs (+ l 2))) ; activation-layer
               (len (array-length (gpu-array ma))) ; number of neurons in current layer
               (eligs (list-ref eliglays el))
               (ei (length eligs))
               (currg (or nextg (gpu-make-matrix ei len)))
               (next2g (gpu-make-matrix len (gpu-cols mw))))
          (if (not nextg) ; begin gradient backward prop using a "fake" error of 1
            (gpu-array-apply currg (lambda (x) 1.))) ; alas, state receives an reinfoce-event
          (gpu-array-apply next2g (lambda (x) 0.))
          ; calculate gradient
          (gpu-refresh mz)
          ;FIX: generalize into: (set-sigmoid-gradient! currg (gpu-array mz))
          (let ((mzarr (gpu-array mz)))
            (do ((i 0 (+ i 1))) ((= i ei)) ; i = foreach elig-mirror / output-neuron
            (do ((j 0 (+ j 1))) ((= j len)) ; j = each neuron in this layer
              (let ((g (array-ref (gpu-array currg) i j))
                    (z (array-ref mzarr j)))
                  (if (= el 0)
                    (if (= i j)
                      (array-set! (gpu-array currg) (* g (ref-sigmoid-grad z)) i j)
                      (array-set! (gpu-array currg) 0. i j))
                    (array-set! (gpu-array currg) (* g (ref-sigmoid-grad z)) i j))))))
          ;----------------------------------------------------------------
          ; update gradient
          (do ((e 0 (+ e 1))) ((= e ei)) ; foreach elig-mirrors
          (do ((i 0 (+ i 1))) ((= i len)) ; foreach neuron in current layer
            (if (or (> el 0) (= i e))
              (gpu-saxpy! (array-ref (gpu-array currg) e i)
                          mx
                          (list-ref eligs e)
                          #f
                          (if (= 0 el) #f i)))))
          ;----------------------------------------------------------------
          ; move gradient to next layer
          (do ((e 0 (+ e 1))) ((= e ei))
          (do ((i 0 (+ i 1))) ((= i len))
            (cblas-saxpy! (array-ref (gpu-array currg) e i)
                          (array-cell-ref (gpu-array mw) i)
                          (array-cell-ref (gpu-array next2g) i))))
          (set! nextg next2g)))))
