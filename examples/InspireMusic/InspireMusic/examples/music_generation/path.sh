# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../:../../third_party/Matcha-TTS:$PYTHONPATH

#!/bin/bash
export MAIN_ROOT=`realpath ${PWD}/../../../`

export PYTHONPATH=${MAIN_ROOT}:${PYTHONPATH}
export BIN_DIR=${MAIN_ROOT}/inspiremusic
