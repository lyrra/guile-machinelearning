
(define (make-net)
  (let ((net (make-array #f 7)))
    (array-set! net (gpu-make-matrix 40 198) 0) ; mhw
    (array-set! net (gpu-make-vector 40)     1) ; vhz
    (array-set! net (gpu-make-vector 40)     2) ; vho
    (array-set! net (gpu-make-matrix 2 40)   3) ; myw
    (array-set! net (gpu-make-vector 2)      4) ; vyz
    (array-set! net (gpu-make-vector 2)      5) ; vyo
    (array-set! net (gpu-make-vector 198)    6) ; vxi
    (array-for-each (lambda (arr)
     (gpu-array-apply arr (lambda (x) (* 0.01 (- (random-uniform) .5)))))
               net)
    net))

(define (net-free net)
  (do ((i 0 (+ i 1))) ((>= i 7))
    (gpu-free-array (array-ref net i))))

(define (net-serialize net)
  (let ((net2 (make-array #f 7)))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (gpu-refresh (array-ref net i))
      (array-set! net2 (gpu-array (array-ref net i)) i))
    net2))

(define (net-make-from arrs)
  (let ((net2 (make-net)))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (gpu-array-copy (array-ref net2 i) (array-ref arrs i)))
    net2))

; get array's as refreshed host-arrays
(define (net-vyo net)
  (let ((rv (array-ref net 5)))
    (gpu-refresh rv)
    (gpu-array rv)))
(define (net-vxi net)
  (let ((rv (array-ref net 6)))
    (gpu-refresh rv)
    (gpu-array rv)))

(define (net-set-input net input)
  (gpu-array-copy (array-ref net 6) input))

(define (net-copy src)
  (let ((dst (make-net)))
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

(define (net-merge! dst src1 src2)
  (array-for-each (lambda (drv vr1 vr2)
                    (array-map! (gpu-array drv)
                                (lambda (a b) (/ (+ a b) 2))
                                (gpu-array vr1) (gpu-array vr2)))
                  dst src1 src2))

(define (sigmoid z)
  (/ 1. (+ 1. (exp (- z)))))

(define (sigmoid-grad z)
  (let ((a (sigmoid z)))
    (* a (- 1 a))))

; Dsigmoid(x) = sigmoid(x) (1 - sigmoid(x))
(define (array-sigmoid src dst)
  (gpu-array-map! dst (lambda (z) (sigmoid z))
                  src))

; calculate gradient GRAD(weight, output)
(define (set-sigmoid-gradient! grad netz)
  (array-map! grad (lambda (z) (sigmoid-grad z))
              netz))

(define (net-run net input)
  (match net
    (#(mhw vhz vho myw vyz vyo vxi)
     (gpu-array-copy vxi input)
     (gpu-sgemv! 1. mhw #f vxi 0. vhz)
     (array-sigmoid vhz vho)
     (gpu-sgemv! 1. myw #f vho 0. vyz)
     (array-sigmoid vyz vyo)
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
