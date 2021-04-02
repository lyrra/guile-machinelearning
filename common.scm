
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
