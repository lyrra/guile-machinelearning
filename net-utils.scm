
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

(define (net-get-stats net)
  (map (lambda (drv)
         (let ((arr (gpu-array drv))
               (n 0)
               (sum 0)
               (mean 0)
               (sum2 0)
               (min #f)
               (max #f))
           (array-for-each (lambda (x)
                             (set! n (1+ n))
                             (set! sum (+ sum x))
                             (if (or (not min) (< x min)) (set! min x))
                             (if (or (not max) (> x max)) (set! max x))
                             )
                           arr)
           (set! mean (/ sum n))
           (array-for-each
            (lambda (x)
              (set! sum2 (+ sum2 (expt (- x mean) 2))))
            arr)
           (list n min max
                 (/ sum n)
                 (/ sum2 n)
                 (sqrt (/ sum2 n)))))
       (list (array-ref (netr-arrs net) 0) ; 0 hidden-layer weights
             (array-ref (netr-arrs net) 3) ; 3 output-layer weights
             )))
