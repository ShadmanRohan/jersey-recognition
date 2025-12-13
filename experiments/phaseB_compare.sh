#!/bin/bash
# Phase B: Compare all attention model results
# Wrapper around unified compare_phase.sh script

cd "$(dirname "$0")"
exec ./compare_phase.sh B

