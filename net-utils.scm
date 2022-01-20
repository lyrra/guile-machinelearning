(define-module (guile-ml net-utils)
  #:use-module (guile-gpu gpu)
  #:use-module (guile-ml common)
  #:use-module (guile-ml net)
  #:use-module (guile-ml rl)
  #:export (randomize-network
            normalize-network
            net-get-stats
            file-load-latest-net))

(define (randomize-network rl conf)
  (let ((randr (get-conf conf 'randr))
        (alpha (get-conf conf 'alpha)))
    (when randr
      (cond
       ((get-conf conf 'rande)
        (let ((f (lambda (layer alpha w e)
                   (+ w (* alpha e randr (- (random-uniform) .5))))))
          (if rl (net-weights-scale (rl-net rl) f alpha))))
       (else
        (let ((f (lambda (layer alpha w e)
                   (+ w (* alpha e randr (- (random-uniform) .5))))))
          (if rl (net-weights-scale (rl-net rl) f alpha))))))))

(define (normalize-network net mag)
  (let ((max 0))
    ; get weight abs-max
    (net-weights-scale net (lambda (layer alpha w e)
                             (if (> (abs w) max)
                                 (set! max (abs w)))
                             w)
                       0)
    (when (> max mag)
      (set! max (/ max mag)) ; dont scale all way down to 1
      (net-weights-scale net (lambda (layer alpha w e)
                               (/ w max))
                         0))))

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

(define* (file-load-latest-net dir #:optional (file-prefix "net-")
                                              (file-suffix ".net"))
  (let ((ds (opendir dir))
        (name #f)
        (episode #f))
    (when (directory-stream? ds)
      (do ((ent (readdir ds) (readdir ds)))
          ((eof-object? ent))
        (if (string-contains ent file-prefix)
          (let* ((as (substring ent (+ (string-length file-prefix) (string-contains ent file-prefix))))
                 (e (string->number (substring as 0 (string-contains as file-suffix)))))
            (when (or (not episode)
                      (> e episode))
              (set! episode e)
              (set! name ent)))))
      (closedir ds))
    (list name episode)))
