;;;; DWIM array operations

(define-module (guile-machinelearning arr)
  #:use-module (ice-9 match)
  #:use-module (common)
  #:use-module (guile-machinelearning mat)
  #:use-module (ml activations)
  #:use-module (ml sigmoid)
  #:export (arr-+ arr-- arr-* arr-/
            arr-+! arr--! arr-*! arr-/!
            vector-map!
            arr-neg
            arr-log
            arr-sumsquare
            arr-dot
            arr-tr
            arr-get-col
            arr-set!
            arr-slice
            arr-select
            arr-insert
            arr-concat-cols
            arr-concat-rows
            arr-apply!
            arr-proc
            arr-zero!
            arr-sigmoid
            arr-tanh
            arr-print))

(define (arr-oper-scalar oper a b new dim reciprocal)
  (if reciprocal
      (cond
       ((= (length dim) 1) ; vector
        (array-index-map! new (lambda (i) (oper a (array-ref b i)))))
       ((= (length dim) 2) ; matrix
        (array-index-map! new (lambda (i j) (oper a (array-ref b i j))))))
      (cond
       ((= (length dim) 1) ; vector
        (array-index-map! new (lambda (i) (oper (array-ref a i) b))))
       ((= (length dim) 2) ; matrix
        (array-index-map! new (lambda (i j) (oper (array-ref a i j) b)))))))

(define-syntax arr-oper!
  (syntax-rules ()
    ((_ a b oper)
       (cond
        ((number? a)
         (let ((dim (array-dimensions b)))
           (arr-oper-scalar oper a b b dim #f)
           b))
        ((number? b)
         (let* ((dim (array-dimensions a)))
           (arr-oper-scalar oper a b a dim #t)
           a))
        (else ; both a and b are arrays
         (let* ((dima (array-dimensions a))
                (dimb (array-dimensions b))
                (ranka (length dima))
                (rankb (length dimb)))
           (cond
            ; simple non-broadcasting vector-vector operation
            ((= ranka rankb 1)
             (let ((r (car dima)))
               (do ((i 0 (1+ i))) ((>= i r))
                 (array-set! a (oper (array-ref a i) (array-ref b i)) i))
               a))
            ((> ranka rankb) ; broadcasting b on a
             (error "cant broadcast"))
            ((< ranka rankb) ; broadcasting a on b
             (error "cant broadcast"))
            (else ; non-broadcasting of two arrays of equal rank
             (let ((da (apply + dima)) (db (apply + dimb)))
               (cond
                ((> da db) ; broadcast b onto a
                 (error "cant broadcast"))
                ((< da db) ; broadcast a onto b
                 (error "cant broadcast"))
                (else 
                 ; non-broadcasting, because ranka = rankb
                 (array-map! a (lambda (a b) (oper a b)) a b)
                 a)))))))))))

(define (arr-+! a b) (arr-oper! a b +))
(define (arr--! a b) (arr-oper! a b -))
(define (arr-*! a b) (arr-oper! a b *))
(define (arr-/! a b) (arr-oper! a b /))

(define (arr-broadcast-oper! oper a b)
  (let* ((dima (array-dimensions a))
         (dimb (array-dimensions b))
         (ranka (length dima))
         (rankb (length dimb))
         (new (if (= ranka rankb 1)
                  (make-vec (car dima))
                  (apply make-arr
                         (cond
                          ((> ranka rankb) dima)
                          ((< ranka rankb) dimb)
                          (else ; equal ranks, take largest dimensions
                           (if (> (apply + dima) (apply + dimb))
                               dima dimb)))))))
    (cond
     ; simple non-broadcasting vector-vector operation
     ((= ranka rankb 1)
      (let ((r (car dima)))
        (do ((i 0 (1+ i))) ((>= i r))
          (array-set! new (oper (array-ref a i) (array-ref b i)) i))
        new))
     ((> ranka rankb) ; broadcasting b on a
      (format #t "broadcasting b on a ranka ~s > rankb ~s~%" dima dimb)
      (error "not-implemented!")
      new)
     ((< ranka rankb) ; broadcasting a on b
      (format #t "broadcasting a on b ranka ~s < rankb ~s~%" dima dimb)
      (error "not-implemented!")
      new)
     (else ; non-broadcasting of two arrays of equal rank
      (let ((da (apply + dima)) (db (apply + dimb)))
        (cond
         ((> da db) ; broadcast b onto a
          (let ()
                                        ; broadcasting b on a, because ranka > rankb
            (cond
             ((= 2 ranka)
              (match (list dima dimb)
                (((xa ya) (xb yb))
                                        ; 5 10
                                        ; 5 1
                 (let ((brc-x (if (= xb 1) 0 #f))
                       (brc-y (if (= yb 1) 0 #f)))
                   (do ((i 0 (1+ i))) ((>= i xa))
                     (do ((j 0 (1+ j))) ((>= j ya))
                       (array-set! new
                                   (oper (array-ref a i j)
                                         (array-ref b
                                                    (or brc-x i)
                                                    (or brc-y j)))
                                   i j))))
                 new)))
             ((= 3 ranka)
              new)
             (else (error "array-oper bad rank" ranka)))))
         ((< da db) ; broadcast a onto b
          (let ()
            ; broadcasting a on b, because ranka < rankb
            (error "not-implemented!")
            new))
         (else 
          ; non-broadcasting, because ranka = rankb
          (array-map! new (lambda (a b) (oper a b)) a b)
          new)))))))


(define (arr-oper-bi! oper a b)
  (cond
   ((and (number? a) (number? b)) (oper a b))
   ((number? a)
    (let* ((dim (array-dimensions b))
           (new (apply make-arr dim)))
      (arr-oper-scalar oper a b new dim #f)
      new))
   ((number? b)
    (let* ((dim (array-dimensions a))
           (new (apply make-arr dim)))
      (arr-oper-scalar oper a b new dim #t)
      new))
   (else ; both a and b are arrays
    (arr-broadcast-oper! oper a b))))

(define (arr-fold! oper acc arrs)
  (if (null? arrs)
      acc
      (arr-fold! oper (arr-oper-bi! oper acc (car arrs)) (cdr arrs))))

(define (arr-oper oper arrs)
  (arr-fold! oper
             (let ((a (car arrs)))
               (if (array? a) (array-copy a) a))
             (cdr arrs)))

(define (arr-+ . arrs) (arr-oper + arrs))
(define (arr-- . arrs) (arr-oper - arrs))
(define (arr-* . arrs) (arr-oper * arrs))
(define (arr-/ . arrs) (arr-oper / arrs))

; same as above but stores result in argument


(define (vector-map! new proc src)
  (let ((n (vector-length new)))
    (do ((i 0 (1+ i)))
        ((>= i n))
      (vector-set! new (proc (vector-ref src i)) i))))

(define-syntax arr-unary-oper
  (syntax-rules ()
    ((_ a x expr)
     (let ((dim (array-dimensions a)))
       (cond
        ((= 1 (length dim))
         (let ((new (make-vec (car dim))))
           (array-map! new
                       (lambda (x) expr)
                       a)
           new))
        (else
         (let ((new (make-arr (car dim) (cadr dim))))
           (array-map! new
                       (lambda (x) expr)
                       a)
           new)))))))

(define (arr-neg a) (arr-unary-oper a x (- x)))
(define (arr-log a) (arr-unary-oper a x (log x)))

(define (arr-sumsquare v)
  (let ((sum 0)
        (n (array-length v)))
    (do ((i 1 (1+ i)))
        ((>= i n))
      (let ((x (array-ref v i)))
        (set! sum (+ sum (* x x)))))
    sum))

(define (arr-dot a b)
  (let* ((dima (array-dimensions a))
         (dimb (array-dimensions b)))
    (cond
     ((= 1 (length dimb))
      (let ((new (make-vec (car dima))))
        (dotv! a b new)
        new))
     (else
      (let ((new (make-arr (car dima) (cadr dimb))))
        (assert (= (cadr dima) (car dimb)))
        (dotm! a b new)
        new)))))

(define (arr-tr a)
  (let ((dima (array-dimensions a)))
    (cond
     ((= (length dima) 1) ; vector -> matrix
      (let ((new (make-arr 1 (car dima)))
            (r (car dima)))
        (do ((i 0 (1+ i)))
            ((>= i r))
          (array-set! new (array-ref a i) 0 i))
        new))
     ((= (length dima) 2)
      (transpose-array a 1 0))
     (else
      ; only wor
      (error "cant transpose a array of dimension" dima)))))

; take column n from A
(define (arr-get-col A n)
  (make-shared-array A
                     (lambda (i) (list i n))
                     (list 0 (1- (car (array-dimensions A))))))

(define (arr-set! A b . inds)
  (let* ((dimb (array-dimensions b))
         (n (car dimb)))
    (do ((i 0 (1+ i)))
        ((>= i n))
      ; FIX: copy to first column (0), use inds
      (array-set! A (array-ref b i) 0 i))))

(define (arr-apply! proc a)
  (array-map! a proc a)
  a)

(define (arr-proc proc . arrs)
  (let ((new (apply make-arr (array-dimensions (car arrs)))))
    (apply array-map! new proc arrs)
    new))
  
(define (arr-slice src idxs)
  (match idxs
    (('* '* z)
     (let* ((dim (array-dimensions src))
            (rows (car dim)) (cols (cadr dim))
            (new (make-arr rows cols)))
       (do ((i 0 (+ i 1))) ((= i rows))
         (do ((j 0 (+ j 1))) ((= j cols))
           (array-set! new (array-ref src i j z) i j)))
       new))
    ((x '* '*) (array-slice src x))
    ((x y '*) (array-slice src x y))))

(define (arr-select src idxs)
  (arr-slice src idxs))

(define (arr-insert dst src idxs)
  (match idxs
    (('* '* z)
     (let* ((dim (array-dimensions src))
            (rows (car dim)) (cols (cadr dim)))
       (do ((i 0 (+ i 1))) ((= i rows))
         (do ((j 0 (+ j 1))) ((= j cols))
           (array-set! dst (array-ref src i j) i j z)))
       dst))))

(define (arr-concat-rows arrs)
  (let ((len 0)
        (rows 0)
        (cols #f))
    (for-each (lambda (arr)
                (match (array-dimensions arr)
                  ((r c)
                   (set! rows (+ rows r))
                   (if (not cols)
                       (set! cols c)
                       (assert (= cols c))))))
              arrs)
    (let ((new (make-arr rows cols))
          (currow 0))
      (for-each (lambda (arr)
                  (match (array-dimensions arr)
                    ((r c)
                     (do ((i 0 (+ i 1))) ((= i r))
                       (do ((j 0 (+ j 1))) ((= j c))
                         (array-set! new (array-ref arr i j)
                                     (+ currow i) j)))
                     (set! currow (+ currow r)))))
                arrs)
      new)))

(define (arr-concat-cols arrs)
  (let ((len 0)
        (rows #f)
        (cols 0))
    (for-each (lambda (arr)
                (match (array-dimensions arr)
                  ((r c)
                   (set! cols (+ cols r))
                   (if (not rows)
                       (set! rows r)
                       (assert (= rows r))))))
              arrs)
    (let ((new (make-arr rows cols))
          (curcol 0))
      (for-each (lambda (arr)
                  (match (array-dimensions arr)
                    ((r c)
                     (do ((i 0 (+ i 1))) ((= i r))
                       (do ((j 0 (+ j 1))) ((= j c))
                         (array-set! new (array-ref arr i j)
                                     i (+ curcol j))))
                     (set! curcol (+ curcol c)))))
                arrs)
      new)))

(define (arr-zero! arr)
  (array-fill! arr 0.0)
  arr)

;;; layer two

(define (arr-sigmoid arr)
  (let ((new (apply make-arr (array-dimensions arr))))
    (array-sigmoid! arr new)
    new))

(define (arr-tanh arr)
  (arr-apply! tanh (array-copy arr)))

(define (arr-print arr . args)
  (match (array-dimensions arr)
    ((r)
     (do ((i 0 (+ i 1))) ((= i r))
       (format #t "~8f " (array-ref arr i)))
     (format #t "~%"))
    ((r c)
     (do ((i 0 (+ i 1))) ((= i r))
       (do ((j 0 (+ j 1))) ((= j c))
         (let ((x (array-ref arr i j)))
           (if (>= x 0)
               (format #t " ~8f " x)
               (format #t "~9f " x))))
       (format #t "~%"))
     (format #t "~%"))
    ((m r c)
     (do ((t 0 (+ t 1))) ((= t m))
       (do ((i 0 (+ i 1))) ((= i r))
         (do ((j 0 (+ j 1))) ((= j c))
           (let ((x (array-ref arr t i j)))
             (if (>= x 0)
                 (format #t " ~8f " x)
                 (format #t "~9f " x))))
         (format #t "~%"))
       (format #t "~%")))
    (_
     (format #t "~s~%" arr))))
