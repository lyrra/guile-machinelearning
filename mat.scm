
(define-module (guile-machinelearning mat)
  #:use-module (ice-9 match)
  #:use-module (guile-machinelearning common)
  #:export (make-arr
            make-vec
            list->arr
            loop-array
            array-copy
            array-zero!
            array-scopy!
            rand-v!
            rand-m!
            fill-v!
            fill-m!
            array-inc!
            dotv! dotm!
            ))

(define (make-vec n)
  (make-typed-array 'f64 *unspecified* n))

(define (make-arr . args)
  (apply make-typed-array 'f64 *unspecified* args))

(define (list->arr lst)
  (let ((rows (length lst))
        (cols (cond
               ((pair? (car lst)) (length (car lst)))
               ((array? (car lst)) (car (array-dimensions (car lst))))
               ((vector? (car lst)) (vector-length (car lst)))
               (else (error "unknown row type:" (car lst))))))
    (let ((arr (make-typed-array 'f64 *unspecified* rows cols)))
      (do ((i 0 (+ 1 i))
           (pair lst (cdr pair)))
          ((null? pair))
        (let ((row (car pair)))
          (cond
           ((pair? row)
            (do ((j 0 (+ 1 j))
                 (r row (cdr r)))
                ((>= j cols))
              (array-set! arr (car r) i j)))
           ((array? (car lst))
            (do ((j 0 (+ 1 j)))
                ((>= j cols))
              (array-set! arr (array-ref row j) i j)))
           ((vector? (car lst))
            (do ((j 0 (+ 1 j)))
                ((>= j cols))
              (array-set! arr (vector-ref row j) i j))))))
      arr)))

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

(define (fill-v! A lst)
  (case (array-type A)
    ((f32 f64)
      (array-index-map! A (lambda (i)
                            (if (>= i (length lst))
                              1.0
                              (list-ref lst i)))))
    (else (throw 'bad-array-type (array-type A))))
  A)

(define (fill-m! A lst)
  (case (array-type A)
    ((f32 f64) (array-index-map!
                A (lambda (i j)
                    (if (>= i (length lst))
                      1.0
                      (if (>= j (length (list-ref lst i)))
                        1.0
                        (list-ref (list-ref lst i) j))))))
    (else (throw 'bad-array-type (array-type A))))
  A)

(define (array-zero! arr)
  (array-map! arr (lambda (x) 0) arr))

(define (array-copy src)
  (let ((dims (array-dimensions src)))
    (let ((dst (apply make-typed-array (append (list 'f32 *unspecified*) dims))))
      (array-map! dst (lambda (x) x) src)
      dst)))

(define (array-scopy! src dst)
  (array-map! dst (lambda (x) x) src))

(define (array-max arr)
  (let ((m #f))
    (array-for-each (lambda (x)
                      (if (or (not m) (> x m))
                          (set! m x)))
                    arr)
    m))

(define* (array-inc! arr pos #:optional v)
  (array-set! arr
              (+ (array-ref arr pos) (or v 1))
              pos))

(define (loop-array fun arr)
  (let ((i 0))
    (array-for-each (lambda (x)
                      (fun i x)
                      (set! i (+ i 1)))
                    arr)))

; y := A*x
(define (dotv! A x y)
  (match (array-dimensions A)
    ((r c)
     (do ((i 0 (+ i 1))) ((= i r))
       (let ((s 0))
         (do ((j 0 (+ j 1))) ((= j c))
           (set! s (+ s (* (array-ref A i j)
                           (array-ref x j)))))
         (array-set! y s i))))))

; y := A*x
(define (dotm! A x y)
  (match (array-dimensions A)
    ((rA cA)
  (match (array-dimensions x)
    ((rx cx)
  (match (array-dimensions y)
    ((ry cy)
     (do ((i 0 (+ i 1))) ((= i rA))
       (do ((j 0 (+ j 1))) ((= j cx))
         (let ((s 0))
           (do ((k 0 (+ k 1))) ((= k rx))
             (set! s (+ s (* (array-ref A i k)
                             (array-ref x k j)))))
           (array-set! y s i j)))))))))))
