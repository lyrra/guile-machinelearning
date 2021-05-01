
(define-record-type <rl>
  (make-rl)
  rl?
  (alpha rl-alpha set-rl-alpha!)
  (gam rl-gam set-rl-gam!)
  (lam rl-lam set-rl-lam!)
  (net rl-net set-rl-net!)
  (Vold rl-Vold set-rl-Vold!)
  (eligs rl-eligs set-rl-eligs!))

(define (new-rl conf net)

  (let* ((arrs (netr-arrs net))
         (numin  (gpu-rows (array-ref arrs 6)))
         (numout (gpu-rows (array-ref arrs 5)))
         (numhid (gpu-rows (array-ref arrs 1)))
         (rl (make-rl)))
    (set-rl-alpha! rl (get-conf conf 'alpha)) ; learning-rate
    (set-rl-gam! rl (get-conf conf 'rl-gam)) ; td-gamma
    (set-rl-lam! rl (get-conf conf 'rl-lam)) ; eligibility-trace decay
    (set-rl-net! rl net)
    (set-rl-Vold! rl (make-typed-array 'f32 0. numout)) ; Vold
    (set-rl-eligs! rl
                   ; eligibility traces, 0-1 is index in output-layer
                   (list (gpu-make-matrix numhid numin) ; mhw-0
                         (gpu-make-matrix numhid numin) ; mhw-1
                         (gpu-make-matrix numout numhid))) ; myw-0
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
         (alpha (rl-alpha rl))
         (gam (rl-gam rl))
         (lam (rl-lam rl))
         (Vnew (net-vyo net))
         (tderr (make-typed-array 'f32 0. (array-length Vold)))
         (vxi (net-vxi net)))
    ;---------------------------------------------
    (cond
     (terminal-state
      ; At the terminal-state we may query our function-approximator,
      ; but often we arent interested in it, because we have access to the final return,
      ; which will ground the us in the truth and flow backwards
      ; ie, tderr <- r + gamma * V(s') - V(s)   at terminal-state should work
      ; but tderr <- r - V(s)   may make more sense
      (sv-! tderr reward Vold)) ; reward - V(s)
     (else
      ; tderr <- r + gamma * V(s') - V(s)
      (svvs*! tderr Vnew gam) ; gamma * V(s')
      (sv-! tderr tderr Vold) ; gamma * V(s') - V(s)
      (sv+! tderr tderr reward) ; R + gamma * V(s') - V(s)
      ; if non-terminal reward is zero and gamma is 1, we can make this short-cut:
      ; (sv-! tderr Vnew Vold)
      ))

    ;---------------------------------------------
    ; discount eligibility traces
    ; update eligibility traces
    ; elig  <- gamma*lambda * elig + Grad_theta(V(s))
    ; z <- y*L* + Grad[V(s,w)]
    (loop-for elig in eligs do
      (gpu-sscal! (* gam lam) elig))
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
         (vxi (net-vxi net)) ; lend networks-input array
         (bvxi (make-typed-array 'f32 *unspecified* (array-length vxi)))
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
  (let* ((net (rl-net rl))
         (numout (gpu-rows (array-ref (netr-arrs net) 5))))
    ; need to rerun network to get fresh output at each layer
    ; needed by backprop
    (net-run net (or (net-vxi net))) ; uses the best-path as input
    (match reward
      ((reward terminal-state)
       ; sane state
       (let ((rewarr (make-typed-array 'f32 0. numout)))
         (cond
          ((>= reward 0)
           (array-set! rewarr 1. 0)
           (array-set! rewarr 0. 1))
          ((< reward 0)
           (array-set! rewarr 0. 0)
           (array-set! rewarr 1. 1)))
         (run-tderr rewarr rl terminal-state))))))
