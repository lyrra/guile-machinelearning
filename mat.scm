
(define-module (guile-machinelearning mat)
  #:use-module (ffi blis arrays) ; from guile-ffi-cblas/mod
  #:use-module (guile-machinelearning common)
  #:export (loop-array
            rand-v!
            rand-m!))

(define (rand-v! A)
  (case (array-type A)
    ((f32 f64)
      (array-index-map! A (lambda (i)
                            (* (- 0.5 (random-uniform)) 0.1))))
    (else (throw 'bad-array-type (array-type A))))
  A)

(define (rand-m! A)
  (case (array-type A)
    ((f32 f64) (array-index-map!
                A (lambda (i j)
                    (* 0.1 (- 0.5 (random-uniform))))))
    (else (throw 'bad-array-type (array-type A))))
  A)
