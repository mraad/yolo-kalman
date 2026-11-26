#!/usr/bin/env bash
# Run project python in isolated mode with a clean LD_LIBRARY_PATH.
# The host LD_LIBRARY_PATH puts /usr/local/cuda/lib (cuDNN 9.20) ahead of the
# venv's cuDNN 9.24 -> CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED on every conv.
cd "$(dirname "$0")"
# Single-threaded inductor compile: its subprocess compile pool intermittently hung when two
# benchmark workers compiled concurrently with a cold cache.
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
exec env -u LD_LIBRARY_PATH .venv/bin/python -I "$@"
