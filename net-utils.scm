
(define (randomize-network rlw rlb conf)
  (let ((randr (get-conf conf 'randr))
        (alpha (get-conf conf 'alpha)))
    (when randr
      (cond
       ((get-conf conf 'rande)
        (let ((f (lambda (layer alpha w e)
                   (+ w (* alpha e randr (- (random-uniform) .5))))))
          (if rlw (net-weights-scale (rl-net rlw) f alpha))
          (if rlb (net-weights-scale (rl-net rlb) f alpha))))
       (else
        (let ((f (lambda (layer alpha w e)
                   (+ w (* alpha e randr (- (random-uniform) .5))))))
          (if rlw (net-weights-scale (rl-net rlw) f alpha))
          (if rlb (net-weights-scale (rl-net rlb) f alpha))))))))

(define (normalize-network net)
  (let ((max 0))
    ; get weight abs-max
    (net-weights-scale net (lambda (layer alpha w e)
                             (if (> (abs w) max)
                                 (set! max (abs w)))
                             w)
                       0)
    ; allow a maximum of magic 5
    (when (> max 1)
      (set! max (/ max 8)) ; dont scale all way down to 1
      (net-weights-scale net (lambda (layer alpha w e)
                               (/ w max))
                         0))))

(define (normalize-networks rlw rlb)
  (if rlw (normalize-network (rl-net rlw)))
  (if rlb (normalize-network (rl-net rlb))))
