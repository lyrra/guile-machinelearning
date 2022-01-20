#!/bin/sh

. ${ENVFILE:-./env.sh}

guile $GUILE_CODE_LOAD_PATH run-tests.scm --gpu $*

