(use-modules (ice-9 binary-ports))
(use-modules (rnrs bytevectors))

(define *verbose* #f)

(define *randstate* #f)

(define* (init-rand #:optional seed)
  (set! *randstate*
        (seed->random-state (or seed (current-time))))
  #f)

(define (random-uniform)
  (random:uniform *randstate*))

(define (random-number len)
  (random len *randstate*))

(define* (assert expr #:optional errmsg)
  (if (not expr)
      (begin
        (if errmsg
          (format #t "  error: ~s~%" errmsg))
        (error "Fatal error."))))

(define-syntax LLL
  (lambda (x)
    (syntax-case x ()
      ((_ e ...)
       #'(if *verbose* (format #t e ...))))))

; fix: default v should be one
(define (array-inc! arr pos v)
  (array-set! arr
              (+ (array-ref arr pos) v)
              pos))

; blas style, consider replacing these with calls into blas
(define (sv-! dst src1 src2)
  (array-map! dst (lambda (a b)
                    (- a b))
              src1 src2))

(define (svvs*! dst vec sc)
  (array-map! dst (lambda (v) (* v sc))
              vec))

(define (get-opt opts key)
  (let ((pair (assq key opts)))
    (if pair (cdr pair) #f)))

(define (indent x)
  (do ((i 0 (1+ i)))
      ((>= i x))
    (format #t " ")))

(define (port-read-uint32 p)
  (let ((bv (make-bytevector 4)))
    (get-bytevector-n! p bv 0 4)
    (bytevector-u32-ref bv 0 (endianness big))))

(define (port-read-float32 p)
  (let ((bv (make-bytevector 4)))
    (get-bytevector-n! p bv 0 4)
    (bytevector-ieee-single-ref bv 0 (endianness big))))

(define (port-write-uint32 p num)
  (let ((bv (make-bytevector 4)))
    (bytevector-u32-set! bv 0 num (endianness big))
    (put-bytevector p bv)))

(define (port-read-array/matrix p)
  (let ((dim (port-read-uint32 p)))
    (cond
     ((= dim 1)
      (let* ((len (port-read-uint32 p))
             (arr (make-array #f len)))
        (do ((i 0 (1+ i)))
            ((>= i len))
          (array-set! arr
                      (port-read-float32 p)
                      i))
        arr))
     ((= dim 2)
      (let* ((rows (port-read-uint32 p))
             (cols (port-read-uint32 p))
             (arr (make-array #f rows cols)))
        (do ((i 0 (1+ i)))
            ((>= i rows))
          (do ((j 0 (1+ j)))
              ((>= j cols))
            (array-set! arr
                        (port-read-float32 p)
                        i j)))
        arr)))))

(define (port-write-array/matrix p arr)
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
      (port-write-uint32 p 1) ; array-dimension
      (port-write-uint32 p arrlen)
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
         (port-write-uint32 p 2) ; array-dimension
         (port-write-uint32 p rows)
         (port-write-uint32 p cols)
         (put-bytevector p bv)))))))
