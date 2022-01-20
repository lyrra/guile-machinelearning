(define-module (guile-ml agent)
  #:use-module (srfi srfi-9)
  #:use-module (guile-ml net)
  #:use-module (guile-ml rl)
  #:use-module (guile-gpu mat)
  #:export (<agent>
            make-agent
            agent?
            agent-net set-agent-net!
            agent-rl  set-agent-rl!
            agent-ovxi set-agent-ovxi!
            new-agent
            agent-init
            agent-end-turn))

(define-record-type <agent>
  (make-agent)
  agent?
  (net agent-net set-agent-net!)
  (rl  agent-rl  set-agent-rl!)
  ; need the previous players input (used at terminal state)
  (ovxi agent-ovxi set-agent-ovxi!))

(define (new-agent net rl)
  (let ((agent (make-agent))
        (numin (if (netr? net) (netr-numin net) 0)))
    (set-agent-net! agent net)
    (set-agent-rl! agent rl)
    (set-agent-ovxi! agent (make-typed-array 'f32 *unspecified* numin))
    agent))

(define (agent-init agent bg transfer-state-net-fun)
  (let* ((net (agent-net agent))
         (rl  (agent-rl agent))
         (vxi (net-vxi net))) ; lend networks-input array
    (rl-episode-clear rl)
    (transfer-state-net-fun bg vxi)
    (net-run net vxi)
    ; Set initial Vold
    (rl-init-step rl)))

; if the same network is used for both players,
; we need to store away the output of the network
; used in the penultimate ply (one half-step before episode terminates),
; this is so the loser-experience can also be learned.
(define (agent-end-turn agent)
  (let ((net (agent-net agent)))
    (if (netr? (agent-net agent))
        (array-scopy! (net-vxi net) (agent-ovxi agent)))))
