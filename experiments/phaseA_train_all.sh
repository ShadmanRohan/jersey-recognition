#!/bin/bash
# Phase A: Train all sequence baseline models (CNN + RNN variants)
# Wrapper around unified train_phase.sh script with discriminative LR

cd "$(dirname "$0")"
exec ./train_phase.sh A 30 64 --use_discriminative_lr --scheduler cosine

