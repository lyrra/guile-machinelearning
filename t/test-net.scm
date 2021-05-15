(define (test-assert-arrays-equal-gpu arv brv epsilon)
  (test-assert-arrays-equal (map (lambda (rv) (gpu-array rv))
                                 (array->list arv))
                            (map (lambda (rv) (gpu-array rv))
                                 (array->list brv))
                            0.0000002))

(define-test (test-net-write/load-arrays)
  (let* ((net (make-net #:in 7 #:out 13 #:hid 53 #:init #t))
         (fname ".tmp-test-arrays")
         (test-array (lambda (arv brv)
           (test-assert-arrays-equal (array->list arv)
                                     (map (lambda (rv) (gpu-array rv))
                                          (array->list brv))
                                     0.0000002))))
    (call-with-output-file fname
      (lambda (p)
        (bio--write-arrays p (netr-arrs net)))
      #:encoding #f #:binary #t)

    (call-with-input-file fname
      (lambda (p)
        (let ((arrs (bio--read-arrays p)))
          (test-array arrs (netr-arrs net))))
      #:guess-encoding #f
      #:encoding #f
      #:binary #t)))

(define-test (test-net-load/save)
  (let* ((in 97)
         (out 13)
         (hid 53)
         (net (make-net #:in in #:out out #:hid hid #:init #t))
         (fname ".tmp-test-net")
         (test-array (lambda (arv brv)
           (test-assert-arrays-equal (map (lambda (rv) (gpu-array rv))
                                          (array->list arv))
                                     (map (lambda (rv) (gpu-array rv))
                                          (array->list brv))
                                     0.0000002))))
    (file-write-net fname 0 net)
    (let ((net2 (file-load-net fname)))
      (test-array (netr-arrs net) (netr-arrs net2)))))

(define-test (test-net-run)
  (let* ((in 2)
         (out 1)
         (hid 2)
         (net (make-net #:in in #:out out #:hid hid #:init #t))
         (arrs (netr-arrs net))
         (vxi (net-vxi net))
         (hw (array-ref arrs 0))
         (yw (array-ref arrs 3))
         (n0w0 0.8) (n0w1 0.4)
         (n1w0 0.2) (n1w1 0.5)
         (n2w0 0.3) (n2w1 0.1)
         ; reference-network
         (refnet (lambda (in)
          (sigmoid-real
           (+ (* n2w0 (sigmoid-real (+ (* n0w0 (array-ref in 0))
                                       (* n0w1 (array-ref in 1)))))
              (* n2w1 (sigmoid-real (+ (* n1w0 (array-ref in 0))
                                       (* n1w1 (array-ref in 1)))))))))
         (compare (lambda (nin)
          (let ((ia (list-ref nin 0))
                (ib (list-ref nin 1)))
            (array-set! vxi (if (eq? ia 'r) (random-uniform) ia) 0)
            (array-set! vxi (if (eq? ia 'r) (random-uniform) ia) 1)
            (net-run net vxi)
            (let* ((vyo (net-vyo net))
                   (a (array-ref vyo 0))
                   (b (refnet vxi))
                   (diff (abs (- a b))))
              (test-assert (epsilon? a b 0.0000001)
                "net-run out=~f diff: ~f, expected out: ~f" a diff b))))))
    ; set weights
    (array-set! (gpu-array hw) n0w0 0 0)
    (array-set! (gpu-array hw) n0w1 0 1)
    (array-set! (gpu-array hw) n1w0 1 0)
    (array-set! (gpu-array hw) n1w1 1 1)
    (array-set! (gpu-array yw) n2w0 0 0)
    (array-set! (gpu-array yw) n2w1 0 1)
    (gpu-dirty-set! hw 1) (gpu-refresh hw)
    (gpu-dirty-set! yw 1) (gpu-refresh yw)
    (loop-for nin in '((0. 0.) (0. 1.) (1. 0.) (1. 1.)
                               (0. .5) (.5 0.) (.5 .5)
                               (0. .1) (.1 0.) (.1 .1)
                               (0. .9) (.9 0.) (.9 .9)) do
      (compare nin))
    (loop-subtests (i) (compare '(r r)))))

(define-test (test-net-update-weights)
  (let* ((in 2)
         (out 1)
         (hid 2)
         (net (make-net #:in in #:out out #:hid hid #:init #t))
         (arrs (netr-arrs net))
         (vxi (net-vxi net))
         (hw (array-ref arrs 0))
         (yw (array-ref arrs 3))
         (alpha 0.1)
         (err 0.4)
         (eps 0.0000001)
         (check-weight (lambda (a b name)
          (test-assert (epsilon? a b eps)
            (format #nil "~a wrong weight, got ~a, expected ~a" name a b)))))
    ; set weights
    (array-set! (gpu-array hw) 0.1 0 0)
    (array-set! (gpu-array hw) 0.3 0 1)
    (array-set! (gpu-array hw) 0.5 1 0)
    (array-set! (gpu-array hw) 0.9 1 1)
    (array-set! (gpu-array yw) 0.6 0 0)
    (array-set! (gpu-array yw) 0.2 0 1)
    (gpu-dirty-set! hw 1) (gpu-refresh hw)
    (gpu-dirty-set! yw 1) (gpu-refresh yw)
    ; run network to ensure update-weights isnt dependent on input/output
    ; only weights + gradients which are invariant to net-run
    (array-set! vxi (random-uniform) 0)
    (array-set! vxi (random-uniform) 1)
    (net-run net vxi)
    ; calculate tderror and back-propagate error
    (let* ((tderr (make-typed-array 'f32 err 1))
           (hgrad (gpu-make-matrix hid in))
           (ograd (gpu-make-vector hid))
           (grads (list (list hgrad)
                        (list ograd))))
      ; manually set gradients (else done by update-eligibility-traces)
      (array-set! (gpu-array hgrad) 0.1 0 0)
      (array-set! (gpu-array hgrad) 0.3 0 1)
      (array-set! (gpu-array hgrad) 0.7 1 0)
      (array-set! (gpu-array hgrad) 0.9 1 1)
      (array-set! (gpu-array ograd) 0.1 0)
      (array-set! (gpu-array ograd) 0.1 1)
      (gpu-dirty-set! hgrad 1)
      (gpu-dirty-set! ograd 1)
      (gpu-refresh hgrad)
      (gpu-refresh ograd)
      (update-weights net alpha tderr grads)
      (gpu-refresh-host hw)
      (gpu-refresh-host yw)
      ; test new weight against old-weight + alpha * tderr * gradient
      (check-weight (array-ref (gpu-array yw) 0 0)
                    (+ 0.6 (* alpha err (array-ref (gpu-array ograd) 0)))
                   "output-neuron-weight-0")
      (check-weight (array-ref (gpu-array yw) 0 1)
                    (+ 0.2 (* alpha err (array-ref (gpu-array ograd) 1)))
                   "output-neuron-weight-1")
      (loop-for parms in '((0 0 0.1) (0 1 0.3) (1 0 0.5) (1 1 0.9)) do
        (match parms
          ((oidx hidx wei)
           (check-weight (array-ref (gpu-array hw) oidx hidx)
                         (+ wei (* alpha err (array-ref (gpu-array hgrad) oidx hidx)))
                         (format #f "hidden-neuron-weight-~a-~a" oidx hidx))))))))
