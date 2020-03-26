;
;
;

(import (srfi srfi-1) (ice-9 match) (srfi srfi-8) (srfi srfi-9))
(import (ffi cblas))
(load "common-lisp.scm")
(load "common.scm")
(load "mat.scm")
(load "backgammon.scm")
(load "td-gammon.scm")

(define *current-test* #f)

(define (test-assert exp . reason)
  (if (not (eq? exp #t))
      (begin
        (format #t "Test ~a has failed. ~s~%" *current-test* reason)
        (exit))))

(load "t/test-backgammon-moves.scm")

(define (main)
  (init-rand)
  (loop-for test in (list test-backgammon-path-1mv) do
    (set! *current-test* test)
    (test))
  #t)

(main)
