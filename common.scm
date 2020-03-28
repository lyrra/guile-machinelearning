
(define *randstate* #f)
(define *rands* #f)

(define (init-rand)
  (set! *rands* (seed->random-state (current-time)))
  (set! *randstate* (seed->random-state (current-time)))
  #f)

(define (random-uniform)
  (random:uniform *randstate*))

(define* (assert expr #:optional errmsg)
  (if (not expr)
      (begin
        (format #t "assert-error!~%")
        (if errmsg
          (format #t "  error: ~s~%" errmsg))
        (exit))))

; fix: default v should be one
(define (array-inc! arr pos v)
  (array-set! arr
              (+ (array-ref arr pos) v)
              pos))
