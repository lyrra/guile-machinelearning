(define-module (guile-machinelearning common-lisp)
  #:export (loop-for
            )
  )

; can't help an old CL'er
(define-syntax loop-for
  (lambda (x)
    (syntax-case x ()
      ((_ x in lst do e e* ...)
       #'(do ((xs lst (cdr xs)))
             ((eq? xs '()))
           (let ((x (car xs)))
             e e* ...))))))

;(let ((lst (list '1 2 3 4))) (loop-for x in lst do (format #t "hej: ~s~%" x)))
;(let ((lst '(1 2 3 4))) (loop-for x in lst do (format #t "hej: ~s~%" x)))
;(loop-for x in (list 1 2 3 4) do (format #t "hej: ~s~%" x))
;(loop-for x in '(1 2 3 4) do (format #t "hej: ~s~%" x))
