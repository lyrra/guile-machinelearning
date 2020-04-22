
(define (make-net)
  (let ((net (make-array #f 7)))
    (array-set! net (rand-m! (make-typed-array 'f32 *unspecified* 40 198)) 0)  ; mhw
    (array-set! net (rand-v! (make-typed-array 'f32 *unspecified* 40)) 1)  ; vhz
    (array-set! net (rand-v! (make-typed-array 'f32 *unspecified* 40)) 2)  ; vho
    (array-set! net (rand-m! (make-typed-array 'f32 *unspecified* 2 40)) 3) ; myw
    (array-set! net (rand-v! (make-typed-array 'f32 *unspecified* 2)) 4)   ; vyz
    (array-set! net (rand-v! (make-typed-array 'f32 *unspecified* 2)) 5)   ; vyo
    (array-set! net (make-typed-array 'f32 *unspecified* 198) 6)           ; vxi
    net))

(define (net-vyo net) (array-ref net 5))
(define (net-vxi net) (array-ref net 6))

(define (net-copy src)
  (let ((dst (make-net)))
    (do ((i 0 (+ i 1))) ((>= i 7))
      (array-map! (array-ref dst i)
                  (lambda (x) x)
                  (array-ref src i)))
    dst))

(define (net-merge! dst src1 src2)
  (array-for-each (lambda (dv v1 v2)
                    (array-map! dv (lambda (a b)
                                     (/ (+ a b) 2))
                                v1 v2))
                  dst src1 src2))

(define (sigmoid z)
  (/ 1. (+ 1. (exp (- z)))))

(define (sigmoid-grad z)
  (let ((a (sigmoid z)))
    (* a (- 1 a))))

; Dsigmoid(x) = sigmoid(x) (1 - sigmoid(x))
(define (array-sigmoid src dst)
  (array-map! dst (lambda (z) (sigmoid z))
              src))

; calculate gradient GRAD(weight, output)
(define (set-sigmoid-gradient! grad netz)
  (array-map! grad (lambda (z) (sigmoid-grad z))
                   netz))

(define (net-run net input)
  (match net
    (#(mhw vhz vho myw vyz vyo vxi)
     (sgemv! 1. mhw CblasNoTrans input 0. vhz)
     (array-sigmoid vhz vho)
     (sgemv! 1. myw CblasNoTrans vho 0. vyz)
     (array-sigmoid vyz vyo)
     #f)))

; gradient-descent, return weight update in grads
(define (update-weights net alpha tderr grads)
  (match grads
    ((emhw0 emhw1 emyw0)
  (match net
    (#(mhw vhz vho myw vyz vyo vxi)
     ; output layer
     (match (array-dimensions myw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each output neuron
          (let ((tde (* alpha (array-ref tderr i))))
            (saxpy! tde (array-cell-ref emyw0 i) (array-cell-ref myw i))))))
     ; hidden layer
     (match (array-dimensions mhw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each hidden neuron
          (let ((tde0 (* alpha (array-ref tderr 0)))
                (tde1 (* alpha (array-ref tderr 1))))
            (saxpy! tde0 (array-cell-ref emhw0 i) (array-cell-ref mhw i))
            (saxpy! tde1 (array-cell-ref emhw1 i) (array-cell-ref mhw i)))))))))))
