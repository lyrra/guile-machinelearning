
(define *current-test* #f)

(define (test-assert exp . reason)
  (if (not (eq? exp #t))
      (begin
        (format #t "Test ~a has failed. ~s~%" *current-test* reason)
        (exit))))

(define (epsilon? a b eps)
  (> eps (abs (- a b))))

(define (test-assert-arrays-equal arrs brrs epsilon)
  ; ensure equal amounts of arrays
  (test-assert (= (length arrs) (length brrs)))
  ; ensure each array is equal dimension
  (for-each (lambda (arr brr)
              (test-assert (equal? (array-dimensions arr)
                                   (array-dimensions brr))))
            arrs brrs)
  ; ensure array content is equal
  (for-each (lambda (arr brr)
              ;(format #t "arr: ~s~%" arr) (format #t "brr: ~s~%" brr)
              (match (array-dimensions arr)
                ((r c)
                 (do ((i 0 (1+ i))) ((= i r))
                 (do ((j 0 (1+ j))) ((= j c))
                   (let ((a (array-ref arr i j))
                         (b (array-ref brr i j)))
                     (test-assert (epsilon? a b epsilon)
                                  (format #f "i:~a,j:~a[~f,~f]" i j a b))))))
                ((r)
                 (do ((i 0 (1+ i))) ((= i r))
                   (let ((a (array-ref arr i))
                         (b (array-ref brr i)))
                     (test-assert (epsilon? a b epsilon)
                                  (format #f "i:~a[~f,~f]" i a b)))))))
            arrs brrs))
