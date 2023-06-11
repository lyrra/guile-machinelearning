;;;; machine-learning
;;;; activation functions
 
(define-module (guile-machinelearning activations)
  #:use-module (ice-9 match)
  #:use-module (guile-machinelearning arr)
  #:use-module (guile-machinelearning mat)
  #:export (; -- softmax --
            softmax softmax-loss
            ; -- sigmoid --
            sigmoid-init
            ref-sigmoid-real
            ref-sigmoid
            ref-sigmoid-grad
            array-sigmoid!
            ))

;;;; softmax

; for numerically stability consider:
; e_x <- np.exp(x - np.max(x))
; e_x / e_x.sum(axis=0)
(define (softmax arr)
  (match (array-dimensions arr)
    ((rows cols)
     (let ((t (array-copy arr))
           (m (make-vec cols)))
       (array-fill! m 0)
       (array-map! t exp t)
       ; summarize over columns
       (do ((i 0 (1+ i))) ((>= i rows))
         (do ((j 0 (1+ j))) ((>= j cols))
           (array-set! m (+ (array-ref m j) (array-ref t i j)) j)))
       ; divide over columns
       (do ((i 0 (1+ i))) ((>= i rows))
         (do ((j 0 (1+ j))) ((>= j cols))
           (let ((s (array-ref m j)))
             (if (not (= s 0))
                 (array-set! t (begin (/ (array-ref t i j) s) ) i j)))))
       t))))

; yh array of outputs from softmax, where softmax outputs a N,1 array
; Y the correct index for each softmax array
; dim Y: i;th output, learning example
(define (softmax-loss yh Y yhi yhj Yi Yj)
  (let ((sum 0)
        (r (car (array-dimensions Y))))
    (match (list yhi yhj Yi Yj)
      ((2 0 0 1)
       (do ((i 0 (1+ i))) ((>= i r))
         (set! sum (- sum
                      (log (array-ref yh (inexact->exact (array-ref Y i 0)) 0 i))))))
      ((0 1 0 1)
       (do ((i 0 (1+ i))) ((>= i r))
         (set! sum (- sum
                      (log (array-ref yh i (inexact->exact (array-ref Y i 0)) 0)))))))
    sum))

;;;; sigmoid


(define *sigmoid-table* #f)

(define (ref-sigmoid-real z) (/ 1. (+ 1. (exp (- z)))))

(define (ref-sigmoid x)
  (let ((i (inexact->exact (truncate (+ (* (/ x 40) 65536) 32768)))))
    (if (< i 0) (set! i 0))
    (if (> i 65535) (set! i 65535))
    (array-ref *sigmoid-table* i)))

(define (ref-sigmoid-grad z)
  (let ((a (ref-sigmoid z)))
    (* a (- 1 a))))

; Dsigmoid(x) = sigmoid(x) (1 - sigmoid(x))
(define (array-sigmoid! src dst)
  (array-map! dst (lambda (z) ;(ref-sigmoid z)
                              (ref-sigmoid-real z)
                              )
                  src))

; calculate gradient GRAD(weight, output)
(define (set-sigmoid-gradient! grad netz)
  (array-map! grad (lambda (z) (ref-sigmoid-grad z))
              netz))

(define (sigmoid-init)
  (set! *sigmoid-table*
         (make-typed-array 'f32 *unspecified* 65536))
  (do ((i 0 (+ i 1)))
      ((= i 65536))
    ;(if (or (< i 10) (> i 65526)) (format #t "~a: ~f~%" i (- (* 40 i (/ 1 65536)) 20)))
    (array-set! *sigmoid-table*
                (ref-sigmoid-real (- (* 40 i (/ 1 65536)) 20))
                i)))

