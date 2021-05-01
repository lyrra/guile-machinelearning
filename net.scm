(use-modules (ice-9 binary-ports))

(define-record-type <netr>
  (make-netr)
  netr?
  (in   netr-in   set-netr-in!)
  (out  netr-out  set-netr-out!)
  (hid  netr-hid  set-netr-hid!)
  (act  netr-act  set-netr-act!)
  (grad netr-grad set-netr-grad!)
  ; old
  (arrs netr-arrs set-netr-arrs!))

(define (file-load-net file)
  (call-with-input-file file
    (lambda (p)
      (port-read-uint32 p) ; version
      (port-read-uint32 p) ; episode
      (let* ((arrlen (port-read-uint32 p))
             (net (make-array #f arrlen)))
        (do ((n 0 (1+ n)))
            ((>= n arrlen))
          (let ((arr (port-read-array/matrix p)))
            (array-set! net arr n)))
        (net-make-from net #f)))
    #:guess-encoding #f
    #:encoding #f
    #:binary #t))

(define (file-write-net file episode net)
  (call-with-output-file file
    (lambda (p)
      (port-write-uint32 p 1) ; version
      (port-write-uint32 p episode)
      (port-write-uint32 p (array-length (netr-arrs net)))
      (array-for-each (lambda (gpu-arr)
                        (gpu-refresh-host gpu-arr)
                        (port-write-array/matrix p (gpu-array gpu-arr)))
                      (netr-arrs net)))
    #:encoding #f #:binary #t))

(define* (make-net #:key (init #t) in out hid)
  (let ((netr (make-netr))
        (net (make-array #f 7)))
    (array-set! net (gpu-make-matrix hid in)  0) ; mhw
    (array-set! net (gpu-make-vector hid)     1) ; vhz
    (array-set! net (gpu-make-vector hid)     2) ; vho
    (array-set! net (gpu-make-matrix out hid) 3) ; myw
    (array-set! net (gpu-make-vector out)     4) ; vyz
    (array-set! net (gpu-make-vector out)     5) ; vyo
    (array-set! net (gpu-make-vector in)      6) ; vxi
    (set-netr-arrs! netr net)
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
         (dst (make-net #:init #f #:in in #:out out #:hid hid)))
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

; gradient-descent, return weight update in grads
(define (update-weights net alpha tderr grads)
  (match grads
    ((emhw0 emhw1 emyw0)
  (match (netr-arrs net)
    (#(mhw vhz vho myw vyz vyo vxi)
     ; output layer
     (match (gpu-array-dimensions myw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each output neuron
          (let ((tde (* alpha (array-ref tderr i))))
            (gpu-saxpy! tde emyw0 myw i i)))))
     ; hidden layer
     (match (gpu-array-dimensions mhw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each hidden neuron
          (let ((tde0 (* alpha (array-ref tderr 0)))
                (tde1 (* alpha (array-ref tderr 1))))
            (gpu-saxpy! tde0 emhw0 mhw i i)
            (gpu-saxpy! tde1 emhw1 mhw i i))))))))))

; gradient-descent, return weight update in grads
(define (update-eligibility-traces net eligs)
  (match eligs
    ((emhw0 emhw1 emyw0)
  (match (netr-arrs net)
    (#(mhw vhz vho myw vyz vyo vxi)
     (let* ((numhid (gpu-rows vhz))
            (go  (make-typed-array 'f32 0.  2))
            (gho (gpu-make-matrix 2 numhid)))
       (gpu-array-apply gho (lambda (x) 0.))
       (gpu-refresh vyz)
       (set-sigmoid-gradient! go (gpu-array vyz))

       (do ((i 0 (+ i 1))) ((= i 2))
         (gpu-saxpy! (array-ref go i) vho emyw0 #f i)
         ;(gpu-saxpy! (array-ref go i) myw gho   i  i)
         (saxpy! (array-ref go i)
                 (array-cell-ref (gpu-array myw) i)
                 (array-cell-ref (gpu-array gho) i)))

       ; gradient through hidden-ouput sigmoid
       ; FIX: make set-sigmoid-gradient! general enough
       (gpu-refresh vhz)
       (let ((vhza (gpu-array vhz)))
       (match (gpu-array-dimensions myw)
         ((r c)
          (do ((i 0 (+ i 1))) ((= i r)) ; i = each output neuron
          (do ((j 0 (+ j 1))) ((= j c)) ; j = each hidden output
            (let ((g (array-ref (gpu-array gho) i j))
                  (z (array-ref vhza j)))
              (array-set! (gpu-array gho) (* g (sigmoid-grad z)) i j)))))))

       (do ((k 0 (+ k 1))) ((= k 2))
         (do ((i 0 (+ i 1))) ((= i numhid))
           (gpu-saxpy! (array-ref (gpu-array gho) k i) vxi (if (= k 0) emhw0 emhw1) #f i)))))))))
