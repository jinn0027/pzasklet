#!/bin/bash

export HF_HOME=/opt/work/hf_cache
pip uninstall -y torchaudio

# --- PEZY-SC3 (pz) デバイス用の環境変数設定 ---
export DEVICE_TYPE_PZ=1
export TORCH_DYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCH_JIT_DISABLE=1
export PYTORCH_JIT=0
export VLLM_USE_TRITON=0
export PZCLExecMode=1
export PZCLHBMInterleave=24
export PZCLVisibleDevices=0,1,2,3

# 無効化フラグの解除
unset VLLM_PZ_DISABLE

python system3.py
