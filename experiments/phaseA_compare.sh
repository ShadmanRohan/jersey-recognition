#!/bin/bash
# Phase A: Compare all sequence baseline model results
# Wrapper around unified compare_phase.sh script

cd "$(dirname "$0")"
exec ./compare_phase.sh A

