#!/bin/bash
# Phase 0: Train all basic models (single-frame control)
# Wrapper around unified train_phase.sh script

cd "$(dirname "$0")"
exec ./train_phase.sh 0 30 64

