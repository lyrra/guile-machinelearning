
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
