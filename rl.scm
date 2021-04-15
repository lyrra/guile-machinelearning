(define %alpha 0.1)

(define-record-type <rl>
  (make-rl)
  rl?
  (gam rl-gam set-rl-gam!)
  (lam rl-lam set-rl-lam!)
  (net rl-net set-rl-net!)
  (Vold rl-Vold set-rl-Vold!)
  (eligs rl-eligs set-rl-eligs!))

(define (new-rl conf net)
  (let ((numhid (gpu-rows (array-ref net 1)))
        (rl (make-rl)))
    (set-rl-gam! rl (get-conf conf 'rl-gam)) ; td-gamma
    (set-rl-lam! rl (get-conf conf 'rl-lam)) ; eligibility-trace decay
    (set-rl-net! rl net)
    (set-rl-Vold! rl (make-typed-array 'f32 0. 2)) ; Vold
    (set-rl-eligs! rl
                   ; eligibility traces, 0-1 is index in output-layer
                   (list (gpu-make-matrix numhid 198) ; mhw-0
                         (gpu-make-matrix numhid 198) ; mhw-1
                         (gpu-make-matrix 2  numhid))) ; myw-0
    rl))

(define (rl-episode-clear rl)
  ; initialize eligibily traces to 0
  (loop-for arr in (rl-eligs rl) do
    (gpu-array-apply arr (lambda (x) 0.))))

(define (rl-init-step rl)
  (let* ((net (rl-net rl))
         (out (net-vyo (rl-net rl))))
    (array-scopy! out (rl-Vold rl))))

; Vold is the previous state-value, V(s), and Vnew is the next state-value, V(s')
(define (run-tderr reward rl terminal-state)
  (let* ((net (rl-net rl))
         (Vold (rl-Vold rl))
         (eligs (rl-eligs rl))
         (gam (rl-gam rl))
         (lam (rl-lam rl))
         (Vnew (net-vyo net))
         (alpha %alpha)
         (tderr (make-typed-array 'f32 0. 2))
         (vxi (net-vxi net)))

    ;---------------------------------------------
    (cond
     (terminal-state
      (sv-! tderr reward Vold)) ; reward - V(s)
     (else
      ;(set! alpha (* alpha 0.1)) ; real reward is worth more (really?)
      ; neural-network version:
      (sv-! tderr Vnew Vold)
      ; table-lookup version:
      ; tderr <- r + gamma * V(s') - V(s)
      ;(svvs*! tderr Vnew gam) ; gamma * V(s')
      ;(sv-! tderr tderr Vold) ; gamma * V(s') - V(s)
      ;(array-map! tderr (lambda (x r) (+ x r)) tderr reward))
      ))

    ;---------------------------------------------
    ; discount eligibility traces
    ; update eligibility traces
    ; elig  <- gamma*lambda * elig + Grad_theta(V(s))
    ; z <- y*L* + Grad[V(s,w)]
    (loop-for elig in eligs do
      (gpu-sscal! lam elig))
    (update-eligibility-traces net eligs)

    ;---------------------------------------------
    ; update network weights (Alpha * Error * Gradient)
    ; delta to update weights: w += alpha * tderr * elig
    ; where elig contains diminished gradients of network activity
    (update-weights net alpha tderr eligs)
    ;(if terminal-state
    ;  (update-weights net alpha tderr eligs))
    ; new net-output becomes old in next step
    (array-scopy! Vnew Vold)))

(define (rl-policy-greedy-action agent cur-state fea-states)
  (let* ((net agent)
         (bvxi (make-typed-array 'f32 *unspecified* 198))
         (vxi (net-vxi net)) ; lend networks-input array
         (points -999)
         (best-state #f))
    (loop-for state in fea-states do
      (set-bg-input state vxi)
      (net-run net vxi)
      (let ((out (net-vyo net)))
        ; FIX: should we consider white(idx-0) > black(idx-1) ?
        (when (> (- (array-ref out 0) (array-ref out 1)) points)
          ; keep best-scored
          ;(LLL "  best-net-out: ~s~%" out)
          (set! points (- (array-ref out 0) (array-ref out 1)))
          (set! best-state state)
          (array-scopy! vxi bvxi))))
    (if best-state ; if path found, ie didn't terminate
        (begin ; restore best-input to network (ie we keep this future)
          (net-set-input net bvxi)
          best-state)
        ; got terminal-state
        #f)))

(define (run-ml-learn bg rl reward)
  (let ((net (rl-net rl)))
    ; need to rerun network to get fresh output at each layer
    ; needed by backprop
    (net-run net (or (net-vxi net))) ; uses the best-path as input
    (match reward
      ((reward terminal-state)
       ; sane state
       (let ((rewarr (make-typed-array 'f32 0. 2)))
         (cond
          ((>= reward 0)
           (array-set! rewarr 1. 0)
           (array-set! rewarr 0. 1))
          ((< reward 0)
           (array-set! rewarr 0. 0)
           (array-set! rewarr 1. 1)))
         (run-tderr rewarr rl terminal-state))))))
