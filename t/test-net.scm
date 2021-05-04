(define (test-assert-arrays-equal-gpu arv brv epsilon)
  (test-assert-arrays-equal (map (lambda (rv) (gpu-array rv))
                                 (array->list arv))
                            (map (lambda (rv) (gpu-array rv))
                                 (array->list brv))
                            0.0000002))

(define-test (test-write/load-arrays)
  (let* ((net (make-net #:in 7 #:out 13 #:hid 53 #:init #t))
         (fname ".tmp-test-arrays")
         (test-array (lambda (arv brv)
           (test-assert-arrays-equal (array->list arv)
                                     (map (lambda (rv) (gpu-array rv))
                                          (array->list brv))
                                     0.0000002))))
    (call-with-output-file fname
      (lambda (p)
        (port-write-arrays p (netr-arrs net)))
      #:encoding #f #:binary #t)

    (call-with-input-file fname
      (lambda (p)
        (let ((arrs (port-read-arrays p)))
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

