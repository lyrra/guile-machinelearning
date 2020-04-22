#!/bin/sh

. ${ENVFILE:-./env.sh}

guile -L $GUILE_CODE_LOAD_PATH run-tests.scm $*

