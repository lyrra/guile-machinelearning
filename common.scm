(use-modules (ice-9 binary-ports))
(use-modules (rnrs bytevectors))

(define *verbose* #f)

(define *randstate* #f)

(define* (init-rand #:optional seed)
  (set! *randstate*
        (seed->random-state (or seed (current-time))))
  #f)

(define (random-uniform)
  (random:uniform *randstate*))

(define (random-number len)
  (random len *randstate*))

(define* (assert expr #:optional errmsg)
  (if (not expr)
      (begin
        (if errmsg
          (format #t "  error: ~s~%" errmsg))
        (error "Fatal error."))))

(define-syntax LLL
  (lambda (x)
    (syntax-case x ()
      ((_ e ...)
       #'(if *verbose* (format #t e ...))))))

; fix: default v should be one
(define (array-inc! arr pos v)
  (array-set! arr
              (+ (array-ref arr pos) v)
              pos))

; blas style, consider replacing these with calls into blas
(define (sv+! dst src1 src2)
  (array-map! dst (lambda (a b)
                    (+ a b))
              src1 src2))
(define (sv-! dst src1 src2)
  (array-map! dst (lambda (a b)
                    (- a b))
              src1 src2))

(define (svvs*! dst vec sc)
  (array-map! dst (lambda (v) (* v sc))
              vec))

(define* (get-conf opts key #:optional (dfl #f))
  (let ((pair (assq key opts)))
    (if pair (cdr pair) dfl)))

(define (set-conf-default conf key val)
  (assq-set! conf key (or (get-conf conf key) val)))

(define (make-conf lst)
  (let ((cfg '()))
    (while #t
      (let ((key (car lst))
            (val (cadr lst)))
        (set! cfg (acons key val cfg))
        (set! lst (cddr lst))
        (if (eq? '() lst) (break))))
    cfg))

(define (merge-conf conf1 conf2)
  (let ((cfg '()))
    (for-each (lambda (pair)
                (match pair
                  ((key . val)
                   (set! cfg (cons (if (assq key conf2)
                                       (assq key conf2)
                                       pair)
                                   cfg)))))
              conf1)
    (for-each (lambda (pair)
                (match pair
                  ((key . val)
                   (if (not (assq key cfg))
                       (set! cfg (cons pair cfg))))))
              conf2)
    cfg))

(define (indent x)
  (do ((i 0 (1+ i)))
      ((>= i x))
    (format #t " ")))

(define (command-line-parse whats)
  (let ((conf '()))
    (do ((args (command-line) (cdr args)))
        ((eq? args '()))
      (let ((arg (car args)))
      (do ((whatp whats (cdr whatp)))
          ((eq? whatp '()))
        (let ((what (car whatp)))
          (match what
            ((name type)
             (cond
              ((eq? type 'string)
               (let ((str (format #f "--~a=" name)))
                 (if (string-contains arg str)
                   (set! conf (acons name (substring arg (string-length str)) conf)))))
              ((eq? type 'boolean)
               (let ((str1 (format #f "--~a" name))
                     (str2 (format #f "--~a=yes" name))
                     (str3 (format #f "--~a=no" name)))
                 (if (string=? arg str1) (set! conf (acons name #t conf)))
                 (if (string=? arg str2) (set! conf (acons name #t conf)))
                 (if (string=? arg str3) (set! conf (acons name #f conf)))))
              ((eq? type 'number)
               (let ((str (format #f "--~a=" name)))
                 (if (string-contains arg str)
                   (set! conf (acons name (string->number (substring arg (string-length str))) conf))))))))))))
    conf))
