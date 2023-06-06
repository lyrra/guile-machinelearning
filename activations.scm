;;;; machine-learning
;;;; activation functions
 
(define-module (ml activations)
  #:use-module (ice-9 match)
  #:use-module (guile-machinelearning arr)
  #:use-module (guile-machinelearning mat)
  #:export (softmax softmax-loss))

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
(define (softmax-loss yh Y . args)
  (let ((selfun (if (null? args) #f (car args))))
    (let ((sum 0)
          (r (car (array-dimensions Y))))
      (if selfun
          (do ((i 0 (1+ i))) ((>= i r))
            (set! sum (- sum (log (selfun i (inexact->exact (array-ref Y i)))))))
          (do ((i 0 (1+ i))) ((>= i r))
            (set! sum (- sum (log (array-ref yh i (inexact->exact (array-ref Y i)) 0))))))
      sum)))
