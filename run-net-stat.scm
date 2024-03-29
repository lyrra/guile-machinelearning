
(import (srfi srfi-1) (ice-9 match) (srfi srfi-8) (srfi srfi-9))
(import (guile-gpu gpu))
(import (guile-gpu sigmoid))
(import (guile-machinelearning common))
(import (guile-machinelearning net))
(import (guile-machinelearning net-utils))

(let ((net-filename #f))
  (do ((args (command-line) (cdr args)))
      ((eq? args '()))
    (if (string-contains (car args) "--net=")
      (set! net-filename (substring (car args) 6))))
  (assert (and net-filename)
          "Usage: --net=<net>")
  (let ((net (file-load-net net-filename)))
    (format #t "in: ~a hid: ~a out: ~a~%"
            (netr-numin net)
            (netr-numhid net)
            (netr-numout net))
    (for-each (lambda (layer-stats)
                (match layer-stats
                  ((n min max avg var std)
                   (format #t "n: ~a [min/max: ~f, ~f] avg: ~f [var: ~f] stddev: ~f~%"
                           n min max avg var std))))
              (net-get-stats net))))

(exit)
