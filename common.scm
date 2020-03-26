
(define *randstate* #f)
(define *rands* #f)

(define (init-rand)
  (set! *rands* (seed->random-state (current-time)))
  (set! *randstate* (seed->random-state (current-time)))
  #f)

(define (random-uniform)
  (random:uniform *randstate*))
