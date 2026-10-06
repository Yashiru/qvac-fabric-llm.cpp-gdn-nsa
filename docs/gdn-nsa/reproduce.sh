#!/usr/bin/env bash
# Reproduce the gdn-nsa results (conversion, correctness, tests, perplexity, performance).
# Run from the repository root. Each step can be run on its own: ./docs/gdn-nsa/reproduce.sh <step>
# Steps: build convert reference compare tests ppl placement bench official
set -euo pipefail

HF_DIR=${HF_DIR:-models/gated-deltanet-nsa-0.4B-10B}   # hf download Idiap/gated-deltanet-nsa-0.4B-10B --local-dir $HF_DIR
GGUF_DIR=${GGUF_DIR:-models/gguf}
WIKI=${WIKI:-data/wikitext-2-raw/wiki.test.raw}
PY=${PY:-python3}                                      # needs torch (CPU is enough), transformers, safetensors, numpy
OUT=${OUT:-gdn-nsa-out}

ROOT=$(pwd)
MC=$ROOT/examples/model-conversion
HF_DIR=$(realpath -m $HF_DIR)
WIKI=$(realpath -m $WIKI)
mkdir -p $OUT && OUT=$(realpath -m $OUT)
F32=$GGUF_DIR/gdn-nsa-0.4B-F32.gguf

step_build() {
    cmake -S . -B build-cpu  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
    cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
    cmake --build build-cpu  -j --target llama-debug llama-perplexity llama-bench llama-completion test-llama-archs
    cmake --build build-cuda -j --target llama-debug llama-perplexity llama-bench llama-completion llama-quantize test-backend-ops
}

step_convert() {
    mkdir -p $GGUF_DIR
    $PY convert_hf_to_gguf.py $HF_DIR --outtype f32 --outfile $F32
    $PY convert_hf_to_gguf.py $HF_DIR --outtype f16 --outfile $GGUF_DIR/gdn-nsa-0.4B-F16.gguf
    ./build-cuda/bin/llama-quantize $F32 $GGUF_DIR/gdn-nsa-0.4B-Q8_0.gguf   Q8_0
    ./build-cuda/bin/llama-quantize $F32 $GGUF_DIR/gdn-nsa-0.4B-Q4_K_M.gguf Q4_K_M
}

# fp32 CPU reference on a 1789-token wikitext prompt (27 complete blocks: selection and window are really sparse)
step_reference() {
    mkdir -p $OUT/ref
    $PY -c "open('$OUT/prompt-long.txt','w',encoding='utf-8').write(open('$WIKI',encoding='utf-8').read()[:6500])"
    (cd $OUT/ref && $PY $MC/scripts/causal/run-org-model-gdn-nsa.py -m $HF_DIR -f $OUT/prompt-long.txt --dump-dir .)
    (cd $OUT/ref && $PY $MC/scripts/causal/run-org-model-gdn-nsa.py -m $HF_DIR -f $OUT/prompt-long.txt --dump-dir . --cast-fp16)
}

# llama.cpp logits of the same prompt: CPU and CUDA, chunked prefill, single prefill, token by token
step_compare() {
    local C="$PY $MC/scripts/causal/compare-logits-gdn-nsa.py"
    local REF=$OUT/ref/data/pytorch-gated-deltanet-nsa-0.4B-10B.bin
    for cfg in "cpu|build-cpu|-ngl 0" "cuda|build-cuda|-ngl 99"; do
        IFS='|' read -r name bdir ngl <<< "$cfg"
        for ub in 512 1; do
            mkdir -p $OUT/lcpp
            ./$bdir/bin/llama-debug -m $F32 -f $OUT/prompt-long.txt --save-logits --logits-output-dir $OUT/lcpp/$name-ub$ub \
                $ngl -ctk f32 -ctv f32 -c 4096 -b 2048 -ub $ub > /dev/null 2>&1
            echo -n "$name ub=$ub (KV F32) vs ref fp32: "; $C $OUT/lcpp/$name-ub$ub/llamacpp-gdn-nsa-0.4B-F32.bin $REF --max-nmse 1e-6
        done
    done
    echo -n "ref cast16 vs ref fp32 (all positions): "
    $C $OUT/ref/pytorch-gated-deltanet-nsa-0.4B-10B-cast16-all-logits.npy $OUT/ref/pytorch-gated-deltanet-nsa-0.4B-10B-all-logits.npy
}

step_tests() {
    ./build-cuda/bin/test-backend-ops test -o NSA_CMP -b CUDA0
    ./build-cuda/bin/test-backend-ops test -o NSA_SEL -b CUDA0
    ./build-cpu/bin/test-llama-archs -a gdn-nsa
}

# NSA needs one sequence per context: -b equal to -c gives n_seq = 1 in llama-perplexity
step_ppl() {
    for b in build-cpu build-cuda; do
        ./$b/bin/llama-perplexity -m $F32 -f $WIKI -c 2048 -b 2048 --chunks 2 -ngl 99 -ctk f32 -ctv f32 2>&1 | grep -E "^\[|Final"
    done
    $PY $MC/scripts/causal/run-org-model-gdn-nsa.py -m $HF_DIR --ppl-file $WIKI --ppl-ctx 2048 --ppl-chunks 2
}

# no silent CPU fallback: count graph nodes per backend
step_placement() {
    GGML_SCHED_DEBUG=2 ./build-cuda/bin/llama-debug -m $F32 -p "Hello, my name is" -ngl 99 -c 4096 -v 2>&1 \
        | awk '/## SPLIT #0:/ { buf = "" } { buf = buf "\n" $0 } END { print buf }' \
        | grep -oE "node #[ 0-9]+\( *[A-Z_0-9]+\): [^[]*\[ *[A-Za-z0-9_]+" \
        | sed -E 's/node #[ 0-9]+\( *([A-Z_0-9]+)\):.*\[ *([A-Za-z0-9_]+)$/\2 \1/' | sort | uniq -c | sort -k2,2 -k1,1rn
}

step_bench() {
    local M="-m $F32 -m $GGUF_DIR/gdn-nsa-0.4B-F16.gguf -m $GGUF_DIR/gdn-nsa-0.4B-Q8_0.gguf -m $GGUF_DIR/gdn-nsa-0.4B-Q4_K_M.gguf"
    ./build-cuda/bin/llama-bench $M -ngl 99 -p 512 -n 128 -r 5
    ./build-cpu/bin/llama-bench  $M -t 6,12,20 -p 512 -n 128 -r 3
    ./build-cuda/bin/llama-bench -m $GGUF_DIR/gdn-nsa-0.4B-F16.gguf -ngl 99 -p 0 -n 64 -d 0,512,1024,2048,3584 -r 3
    ./build-cpu/bin/llama-bench  -m $GGUF_DIR/gdn-nsa-0.4B-F16.gguf -t 6 -p 0 -n 32 -d 0,1024,2048,3584 -r 2
}

# original Triton implementation, on an Ampere or newer GPU (FlashAttention-2 does not support Turing)
step_official() {
    cat << 'EOF'
git clone https://github.com/idiap/hybrid-linear-sparse-attention hlsa   # tested at b786cbc6
python -m venv venv && . venv/bin/activate
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.49.0 einops numpy safetensors huggingface_hub
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
PYTHONPATH=hlsa python examples/model-conversion/scripts/causal/run-org-model-gdn-nsa-official.py \
    -m <hf dir> --tokens-file <run-org-model tokens .bin> --out-dir out --ppl-file wiki.test.raw
EOF
}

for s in "${@:-build convert reference compare tests ppl placement bench}"; do
    for one in $s; do
        echo "=== $one"
        "step_$one"
    done
done
