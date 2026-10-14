# deploy multiple instances of sglang server
set -euxo pipefail

MODEL_PATH="${1:?Usage: bash run_deploy.sh /path/to/model}"

# ===== GCC / C++ compiler =====
GCC_HOME=${GCC_HOME:-/apps/software/spack/gcc/9.3.0/gcc/12.3.0-ad6neh67rioem57tvoynxt24r4yfg354}

export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/g++
export CUDAHOSTCXX=${CXX}
export CMAKE_CUDA_HOST_COMPILER=${CXX}
export NVCC_CCBIN=${CXX}

export PATH=${GCC_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${GCC_HOME}/lib64:${LD_LIBRARY_PATH:-}

# Force nvcc host compiler
export NVCC_PREPEND_FLAGS="-ccbin ${CXX}"

# H100 / sm90
export TORCH_CUDA_ARCH_LIST="9.0"

# Optional: make JIT compile logs easier to locate
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${HOME}/.cache}
mkdir -p "${XDG_CACHE_HOME}"
export OUTLINES_CACHE_DIR=${OUTLINES_CACHE_DIR:-${XDG_CACHE_HOME}/outlines}
mkdir -p "${OUTLINES_CACHE_DIR}"
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/tvm-ffi}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/torch_extensions}

# ===== Debug info =====
echo "===== Compiler check ====="
which gcc
gcc --version
which g++
g++ --version
which nvcc
nvcc --version
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "NVCC_CCBIN=${NVCC_CCBIN}"
echo "NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

# ===== Clear old failed JIT cache =====
if [[ "${CLEAR_JIT_CACHE:-0}" == "1" ]]; then
    rm -rf "${TVM_FFI_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"
fi

TP_SIZE=${TP_SIZE:-1}
NUM_INSTANCES=${NUM_INSTANCES:-2}
PORT_BASE=${PORT_BASE:-23000}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-60000}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-64}
MAX_SINGLE_QUESTION_TOKENS=${MAX_SINGLE_QUESTION_TOKENS:-8192}
HOSTNAME=$(hostname)
mkdir -p logs
declare -a pids=()

cleanup() {
    # Prevent cleanup signals from recursively re-entering this trap.
    trap - INT TERM
    echo "Caught signal, stopping all servers..."
    if (( ${#pids[@]} > 0 )); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# If `CUDA_VISIBLE_DEVICES` is set externally, treat it as the GPU pool and
# shard from it. Otherwise default to 0..(TP_SIZE*NUM_INSTANCES-1).
GPU_POOL_CSV="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "${GPU_POOL_CSV}" ]]; then
    GPU_POOL_CSV="$(seq -s, 0 $((TP_SIZE * NUM_INSTANCES - 1)))"
fi
IFS=',' read -r -a GPU_POOL <<< "${GPU_POOL_CSV}"

required_gpus=$((TP_SIZE * NUM_INSTANCES))
if (( ${#GPU_POOL[@]} < required_gpus )); then
    echo "ERROR: Need at least ${required_gpus} GPUs in CUDA_VISIBLE_DEVICES pool, got ${#GPU_POOL[@]} (${GPU_POOL_CSV})" >&2
    exit 2
fi
echo "GPU pool from parent CUDA_VISIBLE_DEVICES: ${GPU_POOL_CSV}"
echo "Each server sees only its assigned slice, so SGLang logs may show logical cuda:0 even when the physical GPU is not 0."

for (( i=0; i<NUM_INSTANCES; i++ )); do
    gpu_start=$((i * TP_SIZE))
    # Build a comma-separated GPU list from the pool slice.
    gpu_list="${GPU_POOL[gpu_start]}"
    for (( j=1; j<TP_SIZE; j++ )); do
        gpu_list+=",${GPU_POOL[gpu_start + j]}"
    done
    port=$((PORT_BASE + i))

    CUDA_VISIBLE_DEVICES="${gpu_list}" \
    python -m sglang.launch_server \
       --model-path "${MODEL_PATH}" \
       --host 0.0.0.0 \
       --port "${port}" \
       --max-total-tokens "${MAX_TOTAL_TOKENS}" \
       --preferred-sampling-params "{\"max_new_tokens\": ${MAX_SINGLE_QUESTION_TOKENS}}" \
       --tp ${TP_SIZE} \
       --grammar-backend outlines \
       --max-running-requests "${MAX_RUNNING_REQUESTS}" \
       > "logs/${HOSTNAME}_sglang_instance${i}.log" 2>&1 &

    pid=$!
    pids+=("${pid}")
    echo "Launched instance ${i} on physical GPUs ${gpu_list}, port ${port} (PID ${pid})"
done

echo "All ${NUM_INSTANCES} instances launched. Logs: logs/sglang_instance*.log (Ctrl-C to stop)"
wait
