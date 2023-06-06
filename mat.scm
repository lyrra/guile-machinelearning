
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
            make-onehot))

(define (make-vec n)
  (make-typed-array 'f64 *unspecified* n))

(define (make-arr . args)
  (apply make-typed-array 'f64 *unspecified* args))

; guile's version of list->array is kind of strange and less forgiving
(define (list->arr lst)
  (if (null? lst)
      (make-array *unspecified* 0)
      (let ((rows (length lst)))
        (if (number? (car lst)) ; vector
            (let ((arr (make-typed-array 'f64 *unspecified* rows)))
              (do ((i 0 (1+ i))
                   (r lst (cdr r)))
                  ((>= i rows))
                (array-set! arr (car r) i))
              arr)
            (cond
             ((vector? (car lst))
              (let* ((cols (vector-length (car lst)))
                     (arr (make-typed-array 'f64 *unspecified* rows cols)))
                (do ((i 0 (+ 1 i)) (pair lst (cdr pair))) ((null? pair))
                  (do ((j 0 (+ 1 j))) ((>= j cols))

                    (array-set! arr (vector-ref (car pair) j) i j)))
                arr))
             ((array? (car lst))
              (let* ((cols (car (array-dimensions (car lst))))
                     (arr (make-typed-array 'f64 *unspecified* rows cols)))
                (do ((i 0 (+ 1 i)) (pair lst (cdr pair))) ((null? pair))
                  (let ((row (car pair)))
                    (do ((j 0 (+ 1 j))) ((>= j cols))
                      (array-set! arr (array-ref row j) i j))))
                arr))
             ((pair? (car lst))
              (let ((cols (length (car lst))))
                (if (and (> cols 0) (pair? (caar lst))) ; 3D
                    (let* ((depth (length (caar lst)))
                           (arr (make-typed-array 'f64 *unspecified* rows cols depth)))
                      (do ((i 0 (+ 1 i))
                           (pair lst (cdr pair))) ((null? pair))
                        (let ((row (car pair)))
                          (do ((j 0 (+ 1 j))
                               (pair row (cdr pair))) ((null? pair))
                            (let ((col (car pair)))
                              (do ((k 0 (+ 1 k))
                                   (pair col (cdr pair))) ((null? pair))
                                (array-set! arr (car pair) i j k))))))
                      arr)
                    (let ((arr (make-typed-array 'f64 *unspecified* rows cols)))
                       (do ((i 0 (+ 1 i))
                           (pair lst (cdr pair))) ((null? pair))
                        (let ((row (car pair)))
                          (do ((j 0 (+ 1 j))
                               (pair row (cdr pair))) ((null? pair))
                            (array-set! arr (car pair) i j))))
                       arr))))
             (else (error "unknown row type:" (car lst))))))))


(define (rand-v! A)
  (case (array-type A)
    ((f32 f64)
      (array-index-map! A (lambda (i)
                            (* (- 0.5 (random-normal)) 0.1))))
    (else (throw 'bad-array-type (array-type A))))
  A)

(define (rand-m! A)
  (case (array-type A)
    ((f32 f64) (array-index-map!
                A (lambda (i j)
                    (* 0.1 (- 0.5 (random-normal))))))
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

(define (make-onehot X dictsize rowsdim . args)
  (define (aref arr . idxs)
    (let ((x (apply array-ref arr idxs)))
      (if (not (exact? x)) (inexact->exact (round x)) x)))
  (let ((batchdim (if (not (null? args)) (car args) #f))
        (Xdim (array-dimensions X)))
    (if (not batchdim)
      (let* ((rows (list-ref Xdim rowsdim))
             (Xh (make-arr rows dictsize)))
        (array-zero! Xh)
        (do ((i 0 (1+ i))) ((>= i rows))
          (array-set! Xh 1 i (aref X i)))
        Xh)
      (let ((rows (list-ref Xdim rowsdim))
            (batchs (list-ref Xdim batchdim)))
        (cond
         ((= batchdim 0)
          (let ((Xh (make-arr batchs rows dictsize)))
            (array-zero! Xh)
            (do ((b 0 (1+ b))) ((>= b batchdim))
              (do ((i 0 (1+ i))) ((>= i rows))
                (array-set! Xh 1 b i (aref X b i))))
            Xh))
         ((= batchdim 1)
          (let ((Xh (make-arr rows batchs rows dictsize)))
            (array-zero! Xh)
            (do ((b 0 (1+ b))) ((>= b batchdim))
              (do ((i 0 (1+ i))) ((>= i rows))
                (array-set! Xh 1 i b (aref X b i))))
            Xh))
         ((= batchdim 2)
          (let ((Xh (make-arr rows dictsize batchs)))
            (array-zero! Xh)
            (do ((b 0 (1+ b))) ((>= b batchdim))
              (do ((i 0 (1+ i))) ((>= i rows))
                (array-set! Xh 1 i (aref X b i) b)))
            Xh))
         (error "unsupported batchdim" batchdim))))))
