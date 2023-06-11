(define-module (guile-machinelearning examples rnn)
  #:use-module (ice-9 match)
  #:use-module (ice-9 format)
  #:use-module (guile-machinelearning arr)
  #:use-module (guile-machinelearning mat)
  #:use-module (guile-machinelearning activations)
  #:export (rnn-forward rnn-cell-forward
            rnn-backward rnn-cell-backward))

; xt is input at time-step t
; a_prev a_next is the incoming and outgoing activations (recurrent connection)
; n_x is input dimensions (vocabulary-size)
; m is length of mini-batch
; T_x is number of time steps through the RNN
;
; If the training examples consist of different amount of time-steps 
; then T_x will hold the longest training example
;
; input X is a 3D array of dimensions (n_x, m, t)
;
; for each time-step t, take a 2D slice of input, that has all inputs and mini-batches,
; ie array-slice X by (*n_x, *m, t)
;
; The RNN activation has dimensions (n_a, m, T_y)
; and corresponding prediction output, y-hat,has dimension (n_y, m, T_y)

(define (new-arr dim)
  (let ((arr (apply make-array #f dim)))
    (array-fill! arr 0)
    arr))

(define (rnn-cell-forward xt a_prev Waa Wax Wya ba by)
  (let* ((a_next (arr-proc tanh (arr-+ (arr-+ (arr-dot Wax xt) (arr-dot Waa a_prev)) ba)))
         (yt_pred (softmax (arr-+ (arr-dot Wya a_next) by))))
    ;                     FIX: why a_next again?
    (list a_next yt_pred (list a_next a_prev xt))))

(define (rnn-forward x a0 Waa Wax Wya ba by)
  (match (list (array-dimensions x)
               (array-dimensions Wya))
    (((n_x m T_x) (n_y n_a))
     (let (; Create a 3D shape (n_a, m, T_x) of zeros, that holds the hidden state.
           (a (new-arr (list n_a m T_x)))
           ; Create a 3d shape (n_y, m, T_x) that holds predictions.
           (y_pred (new-arr (list n_y m T_x)))
           (a_prev a0)
           (caches '()))
       (do ((t 0 (1+ t)))
           ((>= t T_x))
         (match (rnn-cell-forward (arr-slice x `(* * ,t)) a_prev
                                  Waa Wax Wya ba by)
           ((a_next yt_pred cache)
            (arr-insert a a_next `(* * ,t))
            (arr-insert y_pred yt_pred `(* * ,t))
            (set! caches (cons cache caches))
            (set! a_prev a_next))))
       (list a y_pred
               (list (reverse caches) x))))))

(define (rnn-cell-backward da_next cache Waa Wax Wya ba by)
  (match cache
    ((a_next a_prev xt)
     (let* (; dtanh = (1- np.square(a_next) ) * da_next
            (a_next2 (arr-proc * a_next a_next))
            (dtanh ; computed from a_next and da_next
             (arr-proc * da_next 
                         (arr-proc (lambda (a) (- 1 a)) a_next2)))
            ; gradient of the loss w.r.t Wax
            (dxt (arr-dot (arr-tr Wax) dtanh))     ; dxt = np.dot(Wax.T, dtanh)
            (da_prev (arr-dot (arr-tr Waa) dtanh)) ; da_prev = np.dot(Waa.T, dtanh)
            (dWaa (arr-dot dtanh (arr-tr a_prev))) ; dWaa = np.dot(dtanh, a_prev.T)
            (dWax (arr-dot dtanh (arr-tr xt)))     ; dWax = np.dot(dtanh, xt.T)
            (dba (new-arr (list (car (array-dimensions Waa)) 1))))
       ; dba = np.sum(dtanh, 1, keepdims=True)
       (match (array-dimensions dtanh)
         ((r c)
          (do ((i 0 (1+ i))) ((>= i r))
            (do ((j 0 (1+ j))) ((>= j c))
              (array-set! dba (+ (array-ref dba i 0) (array-ref dtanh i j)) i 0)))))
       (list dxt da_prev dWax dWaa dba)))))

(define (rnn-backward Y yh da caches Waa Wax Wya ba by)
  (let* ((caches-last (car caches))
         (x (cadr caches)))
    (match (car caches-last)
      ((a_next a_prev x1)
       (match (list (array-dimensions yh)
                    (array-dimensions da)
                    (array-dimensions x1))
         (((n_y m T_x) (n_a m2 T_x2) (n_x m3))
          (if (not (= T_x T_x2)) (error "t_x mismatch" T_x T_x2))
          (if (not (= m m2 m3)) (error "m mismatch" m m2 m3))
          (let ((dx   (new-arr (list n_x m T_x)))
                (dWax (new-arr (list n_a n_x)))
                (dWaa (new-arr (list n_a n_a)))
                (dWya (new-arr (list n_y n_a)))
                (dba (new-arr (list n_a 1)))
                (da0 (new-arr (list n_a m)))
                (dby (new-arr (list n_y 1)))
                (da_prevt (new-arr (list n_a m))))
            ; loop through all time steps, T_x
            (let ((d (make-arr n_a m)))
            (do ((t (1- T_x) (1- t))) ((< t 0))
              (do ((n 0 (1+ n))) ((>= n 1))
                ; propagate through softmax
                (let ((dyt (new-arr (list n_y 1)))
                      (idx (inexact->exact (array-ref Y t n)))
                      (a_next (car (list-ref caches-last t))))
                  (do ((i 0 (1+ i))) ((>= i n_y))
                    (array-set! dyt (array-ref yh i 0 t) i 0))
                  (array-set! dyt (- (array-ref dyt idx 0) 1) idx 0)
                  (arr-+! dWya (arr-dot dyt (arr-tr (arr-select a_next `(* ,n) #t))))
                  (arr-+! dby dyt)))
              (array-fill! d 0)
              (arr-+! d (arr-select da `(* * ,t)))
              (arr-+! d da_prevt)
              (match (rnn-cell-backward d (list-ref caches-last t) Waa Wax Wya ba by)
                ((dxt_ da_prev_ dWax_ dWaa_ dba_)
                 (arr-insert dx dxt_ `(* * ,t))
                 (arr-+! dWax dWax_)
                 (arr-+! dWaa dWaa_)
                 (arr-+! dba dba_)
                 (set! da_prevt da_prev_)))))
            (set! da0 da_prevt)
            (list dx da0 dWax dWaa dWya dba dby))))))))
