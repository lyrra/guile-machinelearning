
; gradient-descent, return weight update in grads
(define (update-eligibility-traces net eligs)
  (match eligs
    ((emhw0 emhw1 emyw0)
  (match net
    ((mhw vhz vho myw vyz vyo vxi)
     (let ((go  (make-typed-array 'f32 0.  2))
           (gho (make-typed-array 'f32 0. 2 40)))
       (set-sigmoid-gradient! go vyz)
       (match (array-dimensions myw)
         ((r c)
           (do ((i 0 (+ i 1))) ((= i r)) ; i = each output neuron
             (let ((g (array-ref go i)))
               (do ((j 0 (+ j 1))) ((= j c)) ; j = each hidden output
                 (let* ((o (array-ref vho j))
                        (w (array-ref myw i j))
                        (e (array-ref emyw0 i j)))
              (if (or (> (* g o) 10) (< (* g o) -10)) ; absurd
                  (begin
                   (format #t "emyw0: absurd elig update> e=~f (~a * ~a)~%" (* g o) g o)
                  (exit)))
                   (array-set! emyw0 (+ e (* g o)) i j)
                   (array-set! gho (+ (array-ref gho i j) (* g w)) i j)))))))

       ; gradient through hidden-ouput sigmoid
       ; FIX: make set-sigmoid-gradient! general enough
       (match (array-dimensions myw)
         ((r c)
          (do ((i 0 (+ i 1))) ((= i r)) ; i = each output neuron
          (do ((j 0 (+ j 1))) ((= j c)) ; j = each hidden output
            (let ((g (array-ref gho i j))
                  (z (array-ref vhz j)))
              (array-set! gho (* g (sigmoid-grad z)) i j))))))

       (match (array-dimensions mhw)
         ((r c)
           (do ((k 0 (+ k 1))) ((= k 2)) ; i = each output neuron
             (do ((i 0 (+ i 1))) ((= i r)) ; i = each hidden neuron
               (do ((j 0 (+ j 1))) ((= j c)) ; j = each network-input
                 (let* ((g (array-ref gho k i))
                        (x (array-ref vxi j))
                        (ev (if (= k 0) emhw0 emhw1))
                        (e (array-ref ev i j)))
              (if (or (> (* g x) 10) (< (* g x) -10)) ; absurd
                  (begin
                   (format #t "emhw0/1: absurd elig update> e=~f (~a * ~a)~%" (* g x) g x)
                  (exit)))
                   (array-set! ev (+ e (* g x)) i j)))))))))))))

