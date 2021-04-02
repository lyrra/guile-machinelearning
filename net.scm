(use-modules (ice-9 binary-ports))

(define (file-load-net file)
  (let ((net #f))
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
      #:binary #t)))

(define (file-write-net file episode net)
  (call-with-output-file file
    (lambda (p)
      (port-write-uint32 p 1) ; version
      (port-write-uint32 p episode)
      (port-write-uint32 p (array-length net))
      (array-for-each (lambda (gpu-arr)
                        (gpu-refresh-host gpu-arr)
                        (port-write-array/matrix p (gpu-array gpu-arr)))
                      net))
    #:encoding #f #:binary #t))

(define* (make-net numhid #:optional (init #t))
  (let ((net (make-array #f 7)))
    (array-set! net (gpu-make-matrix numhid 198) 0) ; mhw
    (array-set! net (gpu-make-vector numhid)     1) ; vhz
    (array-set! net (gpu-make-vector numhid)     2) ; vho
    (array-set! net (gpu-make-matrix 2 numhid)   3) ; myw
    (array-set! net (gpu-make-vector 2)      4) ; vyz
    (array-set! net (gpu-make-vector 2)      5) ; vyo
    (array-set! net (gpu-make-vector 198)    6) ; vxi
    (if init
      (array-for-each (lambda (arr)
       (gpu-array-apply arr (lambda (x) (* 0.01 (- (random-uniform) .5)))))
                 net))
    net))

(define (net-free net)
  (do ((i 0 (+ i 1))) ((>= i 7))
    (gpu-free-array (array-ref net i))))

(define (net-serialize net)
  (let ((net2 (make-array #f 7)))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (gpu-refresh-host (array-ref net i))
      (array-set! net2 (gpu-array (array-ref net i)) i))
    net2))

(define* (net-make-from arrs #:optional (init #t))
  (let ((net2 (make-net (array-length (array-ref arrs 1)) init)))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (gpu-array-copy (array-ref net2 i) (array-ref arrs i)))
    net2))

; get array's as refreshed host-arrays
(define (net-vyo net)
  (let ((rv (array-ref net 5)))
    (gpu-refresh-host rv)
    (gpu-array rv)))
(define (net-vxi net)
  (let ((rv (array-ref net 6)))
    (gpu-refresh-host rv)
    (gpu-array rv)))

(define (net-set-input net input)
  (gpu-array-copy (array-ref net 6) input))

(define (net-copy src)
  (let ((dst (make-net (gpu-rows (array-ref src 1)))))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (gpu-array-map! (array-ref dst i)
                      (lambda (x) x)
                      (array-ref src i)))
    dst))

(define (net-transfer dst src)
  (do ((i 0 (+ i 1))) ((>= i 7))
    (let ((srv (gpu-array (array-ref src i)))
          (drv (gpu-array (array-ref dst i))))
      (array-scopy! srv drv))))

(define (net-merge! dst src1 src2 take-ratio)
  (let ((keep-ration (- 1 take-ratio)))
    (array-for-each (lambda (drv vr1 vr2)
                      (array-map! (gpu-array drv)
                                  (lambda (a b)
                                    (+ (* b take-ratio)
                                       (* a keep-ration)))
                                  (gpu-array vr1) (gpu-array vr2)))
                    dst src1 src2)))

(define (net-run net input)
  (match net
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
  (match net
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
  (match net
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
