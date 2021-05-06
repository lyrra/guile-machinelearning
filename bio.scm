; binary input/output
; follows/similar to erlang External Term Format

(use-modules (ice-9 binary-ports))
(use-modules (ice-9 iconv))

(define (bio-read-uint32 p)
  (+ (ash (get-u8 p) 24)
     (ash (get-u8 p) 16)
     (ash (get-u8 p) 8)
     (ash (get-u8 p) 0)))

(define (bio-write-uint32 p n)
  (put-u8 p (ash (logand n #xff000000) -24))
  (put-u8 p (ash (logand n #xff0000) -16))
  (put-u8 p (ash (logand n #xff00) -8))
  (put-u8 p (logand n #xff)))

(define (bio-read-float32 p)
  (let ((bv (make-bytevector 4)))
    (get-bytevector-n! p bv 0 4)
    (bytevector-ieee-single-ref bv 0 (endianness big))))

(define (bio-read-array/matrix p)
  (let ((dim (bio-read-uint32 p)))
    (cond
     ((= dim 1)
      (let* ((len (bio-read-uint32 p))
             (arr (make-array #f len)))
        (do ((i 0 (1+ i)))
            ((>= i len))
          (array-set! arr
                      (bio-read-float32 p)
                      i))
        arr))
     ((= dim 2)
      (let* ((rows (bio-read-uint32 p))
             (cols (bio-read-uint32 p))
             (arr (make-array #f rows cols)))
        (do ((i 0 (1+ i)))
            ((>= i rows))
          (do ((j 0 (1+ j)))
              ((>= j cols))
            (array-set! arr
                        (bio-read-float32 p)
                        i j)))
        arr)))))

(define (bio-write-array/matrix p arr)
  (cond
   ((= 1 (length (array-dimensions arr)))
    (let* ((arrlen (array-length arr))
           (bv (make-bytevector (* 4 arrlen))))
      (do ((i 0 (1+ i)))
          ((>= i arrlen))
        (bytevector-ieee-single-set! bv
                                     (* i 4)
                                     (array-ref arr i)
                                     (endianness big)))
      (bio-write-uint32 p 1) ; array-dimension
      (bio-write-uint32 p arrlen)
      (put-bytevector p bv)))
   ((= 2 (length (array-dimensions arr)))
    (match (array-dimensions arr)
      ((rows cols)
       (let* ((arrlen (* 4 rows cols))
              (bv (make-bytevector arrlen)))
         (do ((i 0 (1+ i)))
             ((>= i rows))
           (do ((j 0 (1+ j)))
               ((>= j cols))
             (bytevector-ieee-single-set! bv
                                          (+ (* j 4) (* cols i 4))
                                          (array-ref arr i j)
                                          (endianness big))))
         (bio-write-uint32 p 2) ; array-dimension
         (bio-write-uint32 p rows)
         (bio-write-uint32 p cols)
         (put-bytevector p bv)))))))

(define (bio-write-f32array p arr)
  (put-u8 p 48)
  (bio-write-array/matrix p arr))

(define (bio-write-list p lst)
  (put-u8 p 108)
  (do ((pair lst (cdr pair)))
      ((eq? '() pair))
    (let ((item (car pair)))
      (bio-write-expr p item)))
  (bio-write-nil p))

(define (bio-write-string p str)
  (put-u8 p 107)
  (bio-write-uint32 p (string-length str))
  (put-bytevector p (string->bytevector str "UTF-8")))

(define (bio-write-symbol p sym)
  (put-u8 p 118)
  (let ((str (symbol->string sym)))
    (bio-write-uint32 p (string-length str))
    (put-bytevector p (string->bytevector str "UTF-8"))))

(define (bio-write-emptylist p) (put-u8 p  1))
(define (bio-write-nil   p)     (put-u8 p  2))
(define (bio-write-false p)     (put-u8 p 49))
(define (bio-write-true  p)     (put-u8 p 50))

(define (bio-write-int p n)
  (put-u8 p 98)
  (bio-write-uint32 p n))

(define (bio-write-expr p expr)
  (cond
    ((eq? '() expr) (bio-write-emptylist p))
    ((eq? #f expr) (bio-write-false p))
    ((eq? #t expr) (bio-write-true  p))
    ((integer? expr) (bio-write-int p expr))
    ((string? expr) (bio-write-string p expr))
    ((list? expr) (bio-write-list p expr))
    ((symbol? expr) (bio-write-symbol p expr))
    (else
      (bio-write-f32array p expr))))

(define (bio-read-string p)
  (let* ((len (bio-read-uint32 p))
         (arr (get-bytevector-n p len)))
    (bytevector->string arr "UTF-8")))

(define (bio-read-list p acc)
  (let ((item (bio-read-expr p)))
    (if (eq? #:nil item)
        (reverse acc)
        (bio-read-list p (cons item acc)))))

(define (bio-read-symbol p)
  (let* ((len (bio-read-uint32 p))
         (arr (get-bytevector-n p len))
         (str (bytevector->string arr "UTF-8")))
    (string->symbol str)))

(define (bio-read-expr p)
  (let ((typ (get-u8 p)))
    (cond
     ((=   1 typ)  '())
     ((=   2 typ)  #:nil)
     ((=  48 typ) (bio-read-array/matrix p))
     ((=  49 typ)  #f)
     ((=  50 typ)  #t)
     ((=  98 typ) (bio-read-uint32 p))
     ((= 107 typ) (bio-read-string p))
     ((= 108 typ) (bio-read-list   p '()))
     ((= 118 typ) (bio-read-symbol p))
     (else
      (error "BIO: unknown type " typ)))))
