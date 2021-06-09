
(define-record-type <rl>
  (make-rl)
  rl?
  (alpha rl-alpha set-rl-alpha!)
  (gam rl-gam set-rl-gam!)
  (lam rl-lam set-rl-lam!)
  (net rl-net set-rl-net!)
  (Vold rl-Vold set-rl-Vold!)
  (eligs rl-eligs set-rl-eligs!)
  (waccu rl-waccu set-rl-waccu!))

(define (new-rl conf net)
  (let ((numin  (netr-numin net))
        (numout (netr-numout net))
        (numhid (netr-numhid net))
        (rl (make-rl)))
    (set-rl-waccu! rl (get-conf conf 'waccu)) ; learning-rate
    (set-rl-alpha! rl (get-conf conf 'alpha)) ; learning-rate
    (set-rl-gam! rl (get-conf conf 'rl-gam)) ; td-gamma
    (set-rl-lam! rl (get-conf conf 'rl-lam)) ; eligibility-trace decay
    (set-rl-net! rl net)
    (set-rl-Vold! rl (make-typed-array 'f32 0. numout)) ; Vold
    (set-rl-eligs! rl #f)
    rl))

(define (rl-episode-clear rl)
  ; in self-play (same network), we need two sets of eligibility-traces
  (if (not (rl-eligs rl))
    (set-rl-eligs! rl (net-grad-clone (rl-net rl))))
  ; initialize eligibily traces to 0
  (net-grad-clear (rl-eligs rl))
  ; clear weight deltas
  (net-wdelta-clear (rl-net rl)))

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
    ;(loop-for elig in eligs do
    ;  (gpu-sscal! (* gam lam) elig))
    (update-eligibility-traces net eligs (* gam lam))

    ;---------------------------------------------
    ; update network weights (Alpha * Error * Gradient)
    ; delta to update weights: w += alpha * tderr * elig
    ; where elig contains diminished gradients of network activity
    (if (rl-waccu rl)
      (net-accu-wdelta net alpha tderr eligs)
      (update-weights net alpha tderr eligs))
    ;
    (if (and terminal-state (rl-waccu rl)) (net-add-wdelta net))
    ; new net-output becomes old in next step
    (array-scopy! Vnew Vold)))

(define (rl-policy-greedy-action agent cur-state fea-states)
  (let* ((net agent)
         (numout (netr-numout net))
         (vxi (net-vxi net)) ; lend networks-input array
         (bvxi (make-typed-array 'f32 *unspecified* (array-length vxi)))
         (points -999)
         (best-state #f))
    (loop-for state in fea-states do
      (set-bg-input state vxi)
      (net-run net vxi)
      (let ((out (net-vyo net)))
        ; FIX: should we consider white(idx-0) > black(idx-1) ?
        (when (> (if (> numout 1)
                     (- (array-ref out 0) (array-ref out 1))
                     (array-ref out 0))
                 points)
          ; keep best-scored
          ;(LLL "  best-net-out: ~s~%" out)
          (set! points (if (> numout 1)
                           (- (array-ref out 0) (array-ref out 1))
                           (array-ref out 0)))
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
         (numout (netr-numout net)))
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
           (if (> numout 1)
             (array-set! rewarr 0. 1)))
          ((< reward 0)
           (array-set! rewarr 0. 0)
           (if (> numout 1)
             (array-set! rewarr 1. 1))))
         (run-tderr rewarr rl terminal-state))))))
