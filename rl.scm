
; [Vold, eligs, gam, lam]
(define (make-rl gam lam)
  (list (make-typed-array 'f32 0. 2) ; Vold
        ; eligibility traces, 0-1 is index in output-layer
        (list (make-typed-array 'f32 *unspecified* 40 198) ; mhw-0
              (make-typed-array 'f32 *unspecified* 40 198) ; mhw-1
              (make-typed-array 'f32 *unspecified* 2 40))  ; myw-0
        gam lam))

(define (rl-episode-clear rl)
  ; initialize eligibily traces to 0
  (match rl
    ((Vold eligs gam lam)
     (loop-for arr in eligs do
       (array-map! arr (lambda (x) 0.) arr)))))

(define (rl-init-step rl net)
  (let ((out (net-vyo net)))
    (match rl
      ((Vold eligs gam lam)
       (scopy! out Vold)))))

; gradient-descent, return weight update in grads
(define (update-eligibility-traces net eligs)
  (match eligs
    ((emhw0 emhw1 emyw0)
  (match net
    (#(mhw vhz vho myw vyz vyo vxi)

     (let ((go  (make-typed-array 'f32 0.  2))
           (gho (make-typed-array 'f32 0. 2 40)))
       (set-sigmoid-gradient! go vyz)

       (do ((i 0 (+ i 1))) ((= i 2))
         (saxpy! (array-ref go i) vho (array-cell-ref emyw0 i))
         (saxpy! (array-ref go i) (array-cell-ref myw i) (array-cell-ref gho i)))

       ; gradient through hidden-ouput sigmoid
       ; FIX: make set-sigmoid-gradient! general enough
       (match (array-dimensions myw)
         ((r c)
          (do ((i 0 (+ i 1))) ((= i r)) ; i = each output neuron
          (do ((j 0 (+ j 1))) ((= j c)) ; j = each hidden output
            (let ((g (array-ref gho i j))
                  (z (array-ref vhz j)))
              (array-set! gho (* g (sigmoid-grad z)) i j))))))

       (do ((k 0 (+ k 1))) ((= k 2))
         (do ((i 0 (+ i 1))) ((= i 40))
           (saxpy! (array-ref (array-cell-ref gho k) i) vxi (array-cell-ref (if (= k 0) emhw0 emhw1) i))))

       ; check gradients aren't crazy
       (array-for-each (lambda (g)
                         (if (or (> g 10) (< g -10)) ; absurd
                             (begin
                              (format #t "emyw0: absurd elig update> g=~f~%" g)
                             (exit))))
                       go)
       (array-for-each (lambda (g)
                         (if (or (> g 10) (< g -10)) ; absurd
                             (begin
                               (format #t "emhw0/1: absurd elig update> g=~f~%" g)
                               (exit))))
                       gho)
       ))))))

; Vold is the previous state-value, V(s), and Vnew is the next state-value, V(s')
(define (run-tderr net reward rl terminal-state)
  (match rl
    ((Vold eligs gam lam)
  (let ((Vnew (net-vyo net))
        (alpha 0.01)
        (tderr (make-typed-array 'f32 0. 2))
        (vxi (net-vxi net)))

    ;---------------------------------------------
    (cond
     (terminal-state
      (sv-! tderr reward Vold)) ; reward - V(s)
     (else
      ; tderr <- r + gamma * V(s') - V(s)
      (svvs*! tderr Vnew gam) ; gamma * V(s')
      (sv-! tderr tderr Vold) ; gamma * V(s') - V(s)
      (array-map! tderr (lambda (x r) (+ x r)) tderr reward)))

    ;---------------------------------------------
    ; update network weights
    ; delta to update weights: w += alpha * tderr * elig
    ; where elig contains diminished gradients of network activity
    ; AEG: alpha * tderr * eligs
    (update-weights net alpha tderr eligs)

    ;---------------------------------------------
    ; discount eligibility traces
    ; update eligibility traces
    ; elig  <- gamma*lambda * elig + Grad_theta(V(s))
    ; z <- y*L* + Grad[V(s,w)]
    (loop-for elig in eligs do
      (matrix-scale! lam elig))
    (update-eligibility-traces net eligs)

    ; new net-output becomes old in next step
    (scopy! Vnew Vold)))))
