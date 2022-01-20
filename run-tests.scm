;
;
;

(import (srfi srfi-1) (ice-9 match) (srfi srfi-8) (srfi srfi-9))
(import (guile-gpu gpu))
(import (guile-ml t runner))

;;; check if gpu is used
(begin
  ;;; check if gpu is used
  (do ((args (command-line) (cdr args)))
      ((eq? args '()))
    (if (string=? (car args) "--gpu")
      (test-env-set #:gpu #t)))
  (cond
   ((test-env? #:gpu)
    (import (guile-gpu rocm-rocblas))
    (gpu-init)
    (gpu-init-thread 0))))

(tests-runner)
