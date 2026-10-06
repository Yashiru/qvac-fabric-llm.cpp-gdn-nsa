#!/usr/bin/env bash
# Reproduce the gdn-nsa results (conversion, correctness, tests, perplexity, performance).
# Run from the repository root. Each step can be run on its own: ./docs/gdn-nsa/reproduce.sh <step>
# Steps: build convert reference compare tests ppl placement bench (default), ppl4096 vulkan checks official (optional)
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
REF=$OUT/ref/data/pytorch-gated-deltanet-nsa-0.4B-10B.bin
CMP="$PY $MC/scripts/causal/compare-logits-gdn-nsa.py"

step_build() {
    cmake -S . -B build-cpu  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
    cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
    cmake --build build-cpu  -j --target llama-debug llama-perplexity llama-bench llama-completion test-llama-archs
    cmake --build build-cuda -j --target llama-debug llama-perplexity llama-bench llama-completion llama-quantize \
        llama-speculative-simple llama-server test-backend-ops test-llama-archs
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
    for cfg in "cpu|build-cpu|-ngl 0" "cuda|build-cuda|-ngl 99"; do
        IFS='|' read -r name bdir ngl <<< "$cfg"
        for ub in 512 1; do
            mkdir -p $OUT/lcpp
            ./$bdir/bin/llama-debug -m $F32 -f $OUT/prompt-long.txt --save-logits --logits-output-dir $OUT/lcpp/$name-ub$ub \
                $ngl -ctk f32 -ctv f32 -c 4096 -b 2048 -ub $ub > /dev/null 2>&1
            echo -n "$name ub=$ub (KV F32) vs ref fp32: "; $CMP $OUT/lcpp/$name-ub$ub/llamacpp-gdn-nsa-0.4B-F32.bin $REF --max-nmse 1e-6
        done
    done
    echo -n "ref cast16 vs ref fp32 (all positions): "
    $CMP $OUT/ref/pytorch-gated-deltanet-nsa-0.4B-10B-cast16-all-logits.npy $OUT/ref/pytorch-gated-deltanet-nsa-0.4B-10B-all-logits.npy
}

step_tests() {
    ./build-cuda/bin/test-backend-ops test -o NSA_CMP -b CUDA0
    ./build-cuda/bin/test-backend-ops test -o NSA_SEL -b CUDA0
    # random-weight gdn-nsa model: CUDA vs CPU logits (NMSE) and GGUF save/load round trip
    ./build-cuda/bin/test-llama-archs -a gdn-nsa
}

# NSA needs one sequence per context: -b equal to -c gives n_seq = 1 in llama-perplexity
step_ppl() {
    for b in build-cpu build-cuda; do
        ./$b/bin/llama-perplexity -m $F32 -f $WIKI -c 2048 -b 2048 --chunks 2 -ngl 99 -ctk f32 -ctv f32 2>&1 | grep -E "^\[|Final"
    done
    $PY $MC/scripts/causal/run-org-model-gdn-nsa.py -m $HF_DIR --ppl-file $WIKI --ppl-ctx 2048 --ppl-chunks 2
}

# training context length (4096), one chunk; the fp32 CPU reference takes several minutes
step_ppl4096() {
    for b in build-cpu build-cuda; do
        echo -n "$b: "; ./$b/bin/llama-perplexity -m $F32 -f $WIKI -c 4096 -b 4096 --chunks 1 -ngl 99 -ctk f32 -ctv f32 2>&1 | grep -oE "Final.*"
    done
    echo -n "fp32 reference: "; $PY $MC/scripts/causal/run-org-model-gdn-nsa.py -m $HF_DIR --ppl-file $WIKI --ppl-ctx 4096 --ppl-chunks 1 | grep -oE "Final.*"
}

# graph nodes per backend of the last graph in a GGML_SCHED_DEBUG=2 log (stdin)
node_counts() {
    awk '/## SPLIT #0:/ { buf = "" } { buf = buf "\n" $0 } END { print buf }' \
        | grep -oE "node #[ 0-9]+\( *[A-Z_0-9]+\): [^[]*\[ *[A-Za-z0-9_]+" \
        | sed -E 's/node #[ 0-9]+\( *([A-Z_0-9]+)\):.*\[ *([A-Za-z0-9_]+)$/\2 \1/' | sort | uniq -c | sort -k2,2 -k1,1rn
}

# no silent CPU fallback: count graph nodes per backend
step_placement() {
    GGML_SCHED_DEBUG=2 ./build-cuda/bin/llama-debug -m $F32 -p "Hello, my name is" -ngl 99 -c 4096 -v 2>&1 | node_counts
}

step_bench() {
    local M="-m $F32 -m $GGUF_DIR/gdn-nsa-0.4B-F16.gguf -m $GGUF_DIR/gdn-nsa-0.4B-Q8_0.gguf -m $GGUF_DIR/gdn-nsa-0.4B-Q4_K_M.gguf"
    ./build-cuda/bin/llama-bench $M -ngl 99 -p 512 -n 128 -r 5
    ./build-cpu/bin/llama-bench  $M -t 6,12,20 -p 512 -n 128 -r 3
    ./build-cuda/bin/llama-bench -m $GGUF_DIR/gdn-nsa-0.4B-F16.gguf -ngl 99 -p 0 -n 64 -d 0,512,1024,2048,3584 -r 3
    ./build-cpu/bin/llama-bench  -m $GGUF_DIR/gdn-nsa-0.4B-F16.gguf -t 6 -p 0 -n 32 -d 0,1024,2048,3584 -r 2
    # one NSA layer at the model shapes, 4096 cached keys: decode and 512-token prefill
    ./build-cuda/bin/test-backend-ops perf -o NSA_CMP -b CUDA0
    ./build-cuda/bin/test-backend-ops perf -o NSA_SEL -b CUDA0
}

# Vulkan has no NSA kernels: the scheduler runs NSA_CMP/NSA_SEL on CPU, visible as extra graph splits
step_vulkan() {
    cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
    cmake --build build-vulkan -j --target llama-debug
    local A="-m $F32 -f $OUT/prompt-long.txt --save-logits -ngl 99 -ctk f32 -ctv f32 -c 4096 -b 2048 -ub 512"
    mkdir -p $OUT/lcpp
    GGML_SCHED_DEBUG=2 ./build-vulkan/bin/llama-debug $A --logits-output-dir $OUT/lcpp/vulkan -v > $OUT/vulkan.log 2>&1
    grep -oE "using device Vulkan0.*|graph splits = [0-9]+" $OUT/vulkan.log | tail -2
    echo "nodes not on Vulkan0:"; node_counts < $OUT/vulkan.log | grep -v " Vulka"   # backend names are truncated to 5 characters
    echo -n "vulkan (KV F32) vs ref fp32: "; $CMP $OUT/lcpp/vulkan/llamacpp-gdn-nsa-0.4B-F32.bin $REF
    # same run with the fp16 paths of the Vulkan backend disabled
    GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 \
        ./build-vulkan/bin/llama-debug $A --logits-output-dir $OUT/lcpp/vulkan-nof16 > /dev/null 2>&1
    echo -n "vulkan, fp16 paths disabled, vs ref fp32: "; $CMP $OUT/lcpp/vulkan-nof16/llamacpp-gdn-nsa-0.4B-F32.bin $REF
}

# other configurations vs the fp32 reference, determinism, state save/load, speculative decoding, guards
step_checks() {
    mkdir -p $OUT/lcpp $OUT/checks
    for cfg in "cpu-kvf16|build-cpu|-ngl 0 -ctk f16 -ctv f16" "cpu-k16v32|build-cpu|-ngl 0 -ctk f16 -ctv f32" \
               "cuda-k16v32|build-cuda|-ngl 99 -ctk f16 -ctv f32" "cuda-k32v16|build-cuda|-ngl 99 -ctk f32 -ctv f16" \
               "cuda-ub37|build-cuda|-ngl 99 -ub 37" "cuda-ub100|build-cuda|-ngl 99 -ub 100" \
               "cuda-ub1789|build-cuda|-ngl 99 -ub 1789" "cuda-run1|build-cuda|-ngl 99" "cuda-run2|build-cuda|-ngl 99" \
               "cpu-t1|build-cpu|-ngl 0 -t 1" "cpu-t7|build-cpu|-ngl 0 -t 7"; do
        IFS='|' read -r name bdir args <<< "$cfg"
        ./$bdir/bin/llama-debug -m $F32 -f $OUT/prompt-long.txt --save-logits --logits-output-dir $OUT/lcpp/$name \
            -ctk f32 -ctv f32 -c 4096 -b 2048 -ub 512 $args > /dev/null 2>&1
        echo -n "$name ($args) vs ref fp32: "; $CMP $OUT/lcpp/$name/llamacpp-gdn-nsa-0.4B-F32.bin $REF
    done
    for pair in "cuda-run1 cuda-run2" "cpu-t1 cpu-t7"; do
        set -- $pair
        cmp -s $OUT/lcpp/$1/llamacpp-gdn-nsa-0.4B-F32.bin $OUT/lcpp/$2/llamacpp-gdn-nsa-0.4B-F32.bin \
            && echo "$1 vs $2: bitwise identical" || echo "$1 vs $2: DIFFERENT"
    done

    # greedy text: plain, then after a full and a partial reuse of a saved session (llama_state_* save/load)
    local P1="$(head -c 1500 $OUT/prompt-long.txt)" P2="$(head -c 1000 $OUT/prompt-long.txt) The film was"
    local G="./build-cuda/bin/llama-completion -m $F32 -ngl 99 -c 2048 --temp 0 -n 24 -no-cnv --no-warmup"
    rm -f $OUT/checks/session.bin
    $G -p "$P1" --prompt-cache $OUT/checks/session.bin > /dev/null 2>&1
    $G -p "$P1" > $OUT/checks/p1-plain.txt 2> /dev/null
    $G -p "$P1" --prompt-cache $OUT/checks/session.bin > $OUT/checks/p1-session.txt 2> /dev/null
    $G -p "$P2" > $OUT/checks/p2-plain.txt 2> /dev/null
    $G -p "$P2" --prompt-cache $OUT/checks/session.bin > $OUT/checks/p2-session.txt 2> $OUT/checks/p2-session.log
    grep -oE "session file matches [0-9]+ / [0-9]+ tokens of prompt" $OUT/checks/p2-session.log || true
    for p in p1 p2; do
        cmp -s $OUT/checks/$p-plain.txt $OUT/checks/$p-session.txt \
            && echo "$p: output with saved session identical" || echo "$p: output with saved session DIFFERENT"
    done

    # speculative decoding, Q4_K_M draft (the hybrid memory cannot drop a suffix: rollback uses checkpoints)
    for p in "Hello, my name is" "The history of the Roman Empire"; do
        ./build-cuda/bin/llama-speculative-simple -m $F32 -md $GGUF_DIR/gdn-nsa-0.4B-Q4_K_M.gguf --spec-type draft-simple \
            -ngl 99 -ngld 99 -c 2048 --temp 0 -n 64 -p "$p" > $OUT/checks/spec.txt 2> $OUT/checks/spec.log
        ./build-cuda/bin/llama-completion -m $F32 -ngl 99 -c 2048 --temp 0 -n 80 -no-cnv --no-warmup -p "$p" \
            > $OUT/checks/nospec.txt 2> /dev/null
        $PY - "$OUT/checks" "$p" "$(grep -oE 'accept *= *[0-9.]+%' $OUT/checks/spec.log)" << 'PYEOF'
import sys
d, p, acc = sys.argv[1:4]
a, b = (open(f"{d}/{f}.txt", encoding="utf-8", errors="replace").read().strip() for f in ("spec", "nospec"))
a, b = a[a.find(p):], b[b.find(p):]
same = b.startswith(a) or a.startswith(b)
print(f"speculative {p!r}: {acc}, {min(len(a), len(b))} chars compared, {'same text as greedy' if same else 'DIFFERENT'}")
PYEOF
    done

    # guards: llama-server defaults to 4 slots (refused, run it with -np 1); quantized KV cache refused
    timeout 60 ./build-cuda/bin/llama-server -m $F32 -ngl 99 --port 18080 2>&1 | grep -m1 -oE "gdn-nsa supports.*" || true
    ./build-cuda/bin/llama-debug -m $F32 -p "Hello" -ngl 99 -ctk q8_0 -ctv q8_0 2>&1 | grep -m1 -oE "gdn-nsa requires.*" || true
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
