
(define (file-load-net file which)
  (let ((net #f))
    (call-with-input-file file
      (lambda (p)
        (let ((x (read p)))
          (set! net
                (car (cdr (if which (caddr x) (cadddr x)))))
          ; if net is saved as a list, convert to vector
          (if (list? net)
              (set! net (list->array 1 net))))))
    (LLL "loaded network!~%")
    net))

(define (file-write-net file episode wnet bnet)
  (call-with-output-file file
    (lambda (p)
      (format p "(#:episode ~a~%" episode)
      (format p "(#:wnet~%")
      (write wnet p)
      (format p "~%)~%(#:bnet~%")
      (write bnet p)
      (format p "))~%"))))

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
     ;----------------------------------------
     (match (array-dimensions myw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each output neuron
          (let ((tde (array-ref tderr i)))
          (do ((j 0 (+ j 1))) ((= j c)) ; j = each hidden output
            (let ((w (array-ref myw i j))
                  (e (array-ref emyw0 i j)))
              (array-set! myw (+ w (* alpha e tde)) i j)))))))
     ; propagate gradient backwards to hidden weights
     (match (array-dimensions mhw)
       ((r c)
        (do ((i 0 (+ i 1))) ((= i r)) ; i = each hidden neuron
          (do ((j 0 (+ j 1))) ((= j c)) ; j = each network-input
            (let ((w (array-ref mhw i j))
                  (e (+ (* (array-ref tderr 0) (array-ref emhw0 i j))
                        (* (array-ref tderr 0) (array-ref emhw0 i j)))))
              (array-set! mhw (+ w (* alpha e)) i j))))))
     #|
     (match (array-dimensions myw)
       ((r c)
        (do ((j 0 (+ j 1))) ((= j c)) ; j = each hidden output
          (let ((w (array-ref myw i j))
                (e (array-ref emyw0 i j)))
            (if (or (> w 10) (< w -10)) ; absurd
                (begin
                 (format #t "absurd weight update> w=~f, e=~f~%" w e)
                 (exit)))))))
     |#
     )))))
