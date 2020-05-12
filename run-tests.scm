;
;
;

(import (srfi srfi-1) (ice-9 match) (srfi srfi-8) (srfi srfi-9))
(import (ffi cblas))
(load "common-lisp.scm")
(load "common.scm")
(load "mat.scm")
(load "sigmoid.scm")
(load "gpu.scm")

;;; check if gpu is used

(let ((gpu #f))
  (do ((args (command-line) (cdr args)))
      ((eq? args '()))
    (if (string=? (car args) "--gpu")
        (set! gpu #t)))
  (cond
   (gpu
    (load "rocm-blas.scm")
    (init-rocblas)
    (init-rocblas-thread 0))))

;;; Load ML/RL

(sigmoid-init)
(load "net.scm")
(load "rl.scm")
(load "backgammon.scm")
(load "td-gammon.scm")

(load "t/test-common.scm") ; test driver
;;; tests
(load "t/test-blas.scm")
(load "t/test-cblas.scm")
(load "t/test-cblas-blas.scm")
(load "t/test-rocm-blas.scm")
(load "t/test-gpu-rocm-net.scm")
(load "t/test-backgammon-moves.scm")

(define (main)
  (init-rand)
  (loop-for test in (list test-blas-copy
                          test-blas-sscal
                          test-blas-saxpy
                          test-blas-saxpy-2
                          test-cblas-blas-sgemv
                          test-cblas-blas-saxpy
                          test-rocm-blas-saxpy
                          test-rocm-blas-sgemv
                          test-gpu-rocm-sigmoid
                          test-gpu-rocm-net
                          test-backgammon-valid-pos
                          test-backgammon-bar-pos
                          test-backgammon-path-edge
                          test-backgammon-path-1mv test-backgammon-path-2mv
) do
    (set! *current-test* test)
    (test))
  #t)

(main)
