
(define (file-load-net file which)
  (let ((net #f))
    (call-with-input-file file
      (lambda (p)
        (let ((x (read p)))
          (set! net
                (car (cdr (if which (caddr x) (cadddr x))))))))
    (LLL "loaded network!~%")
    net))

(define (make-net)
  (let ((mhw (rand-m! (make-typed-array 'f32 *unspecified* 40 198)))
        (vhz (rand-v! (make-typed-array 'f32 *unspecified* 40)))
        (vho (rand-v! (make-typed-array 'f32 *unspecified* 40)))
        (myw (rand-m! (make-typed-array 'f32 *unspecified* 2 40)))
        (vyz (rand-v! (make-typed-array 'f32 *unspecified* 2)))
        (vyo (rand-v! (make-typed-array 'f32 *unspecified* 2)))
        (vxi (make-typed-array 'f32 *unspecified* 198)))
    (list mhw vhz vho myw vyz vyo vxi)))

(define (net-vyo net) (list-ref net 5))
(define (net-vxi net) (list-ref net 6))

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
    ((mhw vhz vho myw vyz vyo vxi)
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
    ((mhw vhz vho myw vyz vyo vxi)
     ;----------------------------------------
     (match (array-dimensions myw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each output neuron
          (let ((tde (array-ref tderr i)))
          (do ((j 0 (+ j 1))) ((= j c)) ; j = each hidden output
            (let ((w (array-ref myw i j))
                  (e (array-ref emyw0 i j)))
              (array-set! myw (+ w (* alpha e tde)) i j)
              (if (or (> w 10) (< w -10)) ; absurd
                  (begin
                   (format #t "absurd weight update> w=~f, e=~f~%" w e)
                  (exit)))))))))
     ; propagate gradient backwards to hidden weights
     (match (array-dimensions mhw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each hidden neuron
          (do ((j 0 (+ j 1))) ((= j c)) ; j = each network-input
            (let ((w (array-ref mhw i j))
                  (e (+ (* (array-ref tderr 0) (array-ref emhw0 i j))
                        (* (array-ref tderr 0) (array-ref emhw0 i j)))))
              (array-set! mhw (+ w (* alpha e)) i j)))))))))))
