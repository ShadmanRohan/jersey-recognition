#!/bin/bash
# Phase B: Train all attention models (Lightweight Attention / Frame Selection)
# Wrapper around unified train_phase.sh script with discriminative LR

cd "$(dirname "$0")"
exec ./train_phase.sh B 30 64 --use_discriminative_lr --scheduler cosine

