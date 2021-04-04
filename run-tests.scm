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

(load "softmax.scm")
(sigmoid-init)
(load "net.scm")
(load "rl.scm")
(load "agent.scm")
(load "backgammon.scm")
(load "td-gammon.scm")

(load "t/test-common.scm") ; test driver
;;; tests
(load "t/test-softmax.scm")
(load "t/test-blas.scm")
(load "t/test-cblas.scm")
(load "t/test-cblas-blas.scm")
(load "t/test-gpu.scm") ; test gpu-abstraction without hardware
(load "t/test-rocm-blas.scm") ; test gpu-abstraction with hardware
(load "t/test-gpu-rocm-net.scm")
(load "t/test-backgammon-moves.scm")

(begin
  (init-rand)
  (run-tests '(test-softmax
               test-blas-copy
               test-blas-sscal
               test-blas-saxpy
               test-blas-saxpy-2
               test-cblas-blas-sgemv
               test-cblas-blas-saxpy
               test-rocm-blas-saxpy
               test-rocm-blas-sgemv
               test-gpu-array-copy
               test-gpu-sscal
               test-gpu-rocm-sigmoid
               test-gpu-rocm-net
               test-backgammon-valid-pos
               test-backgammon-bar-pos
               test-backgammon-path-edge
               test-backgammon-path-1mv test-backgammon-path-2mv
               test-backgammon-path-dual)))
