#!/bin/bash
# Phase 0: Compare all basic model results
# Wrapper around unified compare_phase.sh script

cd "$(dirname "$0")"
exec ./compare_phase.sh 0

