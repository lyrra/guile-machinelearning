
(define-test (test-net-load/save)
  (let* ((in 97)
         (out 7)
         (hid 53)
         (net (make-net #:in in #:out out #:hid hid))
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
