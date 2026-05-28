#!/usr/bin/env bash
# Reproduce InferenceX dsv4-fp4-mi355x-vllm conc=128 benchmark locally.
set -eo pipefail

ROOT="/shared/amdgpu/home/fai_qle/vllm"
MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
PORT="${PORT:-8001}"
CONC="${CONC:-128}"
NUM_PROMPTS="${NUM_PROMPTS:-$((CONC * 10))}"
TAG="${TAG:-baseline}"
RESULT_DIR="${ROOT}/bench_results/dsv4_conc128_repro/${TAG}"
SERVER_LOG="${RESULT_DIR}/server.log"
mkdir -p "${RESULT_DIR}"

export HF_HOME="${HF_HOME:-/shared/amdgpu/home/fai_qle/scratch/models}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export VLLM_ROCM_USE_AITER=1
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [[ "${DISABLE_DSV4_CAP:-0}" == "1" ]]; then
  export VLLM_ROCM_DSV4_DEFAULT_MAX_NUM_SEQS=0
else
  unset VLLM_ROCM_DSV4_DEFAULT_MAX_NUM_SEQS
fi

cd "${ROOT}"
source .venv/bin/activate

if [[ "${START_SERVER:-1}" == "1" ]]; then
  pkill -f "vllm serve ${MODEL}" 2>/dev/null || true
  sleep 2
  echo "Starting server (cap disabled=${DISABLE_DSV4_CAP})..."
  "${ROOT}/.venv/bin/vllm" serve "${MODEL}" --port "${PORT}" \
    --tensor-parallel-size 8 \
    --async-scheduling \
    --no-enable-prefix-caching \
    --distributed-executor-backend mp \
    --gpu-memory-utilization 0.8 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --moe-backend triton_unfused \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  echo "${SERVER_PID}" > "${RESULT_DIR}/server.pid"

  for _ in $(seq 1 720); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "Server ready"
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Server died"; tail -50 "${SERVER_LOG}"; exit 1
    fi
    sleep 10
  done
fi

BENCH_DIR="/shared/amdgpu/home/fai_qle/InferenceX/utils/bench_serving"
RANGE_RATIO="${RANGE_RATIO:-0.8}"
PYTHONPATH="${BENCH_DIR}:${ROOT}" \
  .venv/bin/python "${BENCH_DIR}/benchmark_serving.py" \
  --backend vllm \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "${MODEL}" \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --random-range-ratio "${RANGE_RATIO}" \
  --num-prompts "${NUM_PROMPTS}" \
  --max-concurrency "${CONC}" \
  --ignore-eos \
  --trust-remote-code \
  --dsv4 \
  --use-chat-template \
  --result-dir "${RESULT_DIR}" \
  --result-filename "bmk_conc${CONC}.json" \
  --num-warmups $((CONC * 2)) \
  2>&1 | tee "${RESULT_DIR}/bench.log"
