(define-module (guile-machinelearning rl)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 match)
  #:use-module (guile-machinelearning common-lisp)
  #:use-module (guile-machinelearning common)
  #:use-module (guile-machinelearning net)
  #:use-module (guile-machinelearning agent)
  #:use-module (guile-gpu mat)
  #:export (<rl>
            make-rl
            rl?
            rl-alpha set-rl-alpha!
            rl-gam set-rl-gam!
            rl-lam set-rl-lam!
            rl-net set-rl-net!
            rl-Vold set-rl-Vold!
            rl-eligs set-rl-eligs!
            rl-waccu set-rl-waccu!
            new-rl
            rl-episode-clear
            rl-init-step
            run-tderr
            rl-policy-greedy-action
            rl-policy-greedy-action-topn
            run-ml-learn))

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
  (if (netr-wdelta (rl-net rl))
    (net-wdelta-clear (rl-net rl))))

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

(define (rl-policy-greedy-action agent cur-state fea-states transfer-state-net-fun)
  (let* ((net agent)
         (numout (netr-numout net))
         (vxi (net-vxi net)) ; lend networks-input array
         (bvxi (make-typed-array 'f32 *unspecified* (array-length vxi)))
         (points -999)
         (best-state #f))
    (loop-for state in fea-states do
      (transfer-state-net-fun state vxi)
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

; like rl-policy-greedy-action but
; returns the nth-best action at position topn
(define (rl-policy-greedy-action-topn agent cur-state fea-states transfer-state-net-fun topn)
  ;(format #t "rl-policy-greedy-action-topn, select topn=~s~%" topn)
  (let* ((net agent)
         (numout (netr-numout net))
         (vxi (net-vxi net)) ; lend networks-input array
         (bvxi (make-typed-array 'f32 *unspecified* (array-length vxi)))
         (bests (make-array #f topn)) ; vector of best-state
         (bestp (make-array #f topn))) ; vector of best-score
    (loop-for state in fea-states do
      (transfer-state-net-fun state vxi)
      (net-run net vxi)
      (let* ((out (net-vyo net))
             (score (if (> numout 1)
                        (- (array-ref out 0) (array-ref out 1))
                        (array-ref out 0)))
             (pushs #f)
             (pushp #f))
          (cond
           ; no state at this slot
           ((not (array-ref bests 0))
            (array-set! bests state 0)
            (array-set! bestp score 0))
           ; state is better than state in this slot
           ((> score (array-ref bestp 0))
            (set! pushs (array-ref bests 0))
            (set! pushp (array-ref bestp 0))
            (array-set! bests state 0)
            (array-set! bestp score 0)))
          (if pushs ; found a better state, rotate old states
            (do ((i 0 (1+ i)))
                ((>= i topn))
              (if (> i 0)
                (let ((os (array-ref bests i))
                      (op (array-ref bestp i)))
                  (array-set! bests pushs i)
                  (array-set! bestp pushp i)
                  (set! pushs os)
                  (set! pushp op)))))))
    (cond
     ((array-ref bests 0)
      (let ((state (array-ref bests 0)))
        (transfer-state-net-fun state bvxi)
        (net-set-input net bvxi)
        ; FIX: responsibility to restore networks transitories
        ;(net-run net vxi)
        state))
     (else ; terminal-state
      #f))))

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
          ((> reward 0)
           (array-set! rewarr 1. 0)
           (if (> numout 1)
             (array-set! rewarr 0. 1)))
          ((<= reward 0)
           (array-set! rewarr 0. 0)
           (if (> numout 1)
             (array-set! rewarr 1. 1))))
         (run-tderr rewarr rl terminal-state))))))
