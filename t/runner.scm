
(define-module (guile-machinelearning t runner)
  #:use-module (ice-9 match)
  #:use-module (ice-9 format)
  #:use-module (guile-machinelearning common)
  #:use-module (guile-machinelearning common-lisp)
  #:use-module (guile-gpu gpu)
  #:use-module (guile-gpu sigmoid)
  #:use-module (guile-gpu mat)
  #:use-module (guile-machinelearning mat)
  #:use-module (guile-machinelearning bio)
  #:use-module (guile-machinelearning net)
  #:use-module (guile-machinelearning rl)
  #:use-module (guile-machinelearning t test-common)
  #:export (tests-runner)
  #:re-export (test-env-set test-env?))

(define (tests-runner)

  (sigmoid-init)
  (init-rand)

  ;;; Load ML/RL
  (sigmoid-init)

  (set-current-module (resolve-module '(guile-ml t runner)))
  (format #t "running tests in scheme-module: ~s~%" (current-module))
  ;;; tests
  (load "test-gpu-rocm-net.scm")
  (load "test-net.scm")
  (load "test-bio.scm")

  (run-tests))
