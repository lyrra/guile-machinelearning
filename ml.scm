(define-module (guile-ml ml)
  #:use-module (guile-ml bio)
  #:use-module (guile-ml common)
  #:use-module (guile-ml net)
  #:use-module (guile-ml net-utils)
  #:use-module (guile-ml rl)
  #:use-module (guile-ml agent)
  #:re-export (; bio
               bio-read-uint32
               bio-write-uint32
               bio-read-float32
               bio-read-array/matrix
               bio-write-array/matrix
               bio-write-f32array
               bio-write-list
               bio-write-string
               bio-write-symbol
               bio-write-emptylist
               bio-write-nil
               bio-write-false
               bio-write-true
               bio-write-int
               bio-write-expr
               bio-read-string
               bio-read-list
               bio-read-symbol
               bio-read-expr
               ; common
               assert
               init-rand
               random-uniform
               random-number
               get-conf
               set-conf-default
               make-conf
               merge-conf
               indent
               command-line-parse
               ; net
               <netr>
               make-net
               netr?
               netr-info set-netr-info!
               netr-numin set-netr-numin!
               netr-numout set-netr-numout!
               netr-numhid set-netr-numhid!
               netr-in   set-netr-in!
               netr-out  set-netr-out!
               netr-hid  set-netr-hid!
               netr-act  set-netr-act!
               netr-grad set-netr-grad!
               netr-arrs set-netr-arrs!
               netr-wdelta set-netr-wdelta!
               net-grad-clone
               net-grad-clear
               net-wdelta-clear
               net-vyo
               net-vxi
               update-eligibility-traces
               net-weights-scale
               net-make-wdelta
               net-accu-wdelta
               update-weights
               net-add-wdelta
               net-run
               net-set-input
               net-copy
               net-merge!
               net-transfer
               file-write-net
               file-load-net
               ; net-utils
               randomize-network
               normalize-network
               net-get-stats
               file-load-latest-net
               ; agent
               <agent>
               make-agent
               agent?
               agent-net set-agent-net!
               agent-rl  set-agent-rl!
               agent-ovxi set-agent-ovxi!
               new-agent
               agent-init
               agent-end-turn
               ; rl
               <rl>
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
