#!/usr/bin/env bash

# LLAMA_CLI=/mnt/playground/sudeep/llama.cpp-rknn-debug/build_release/bin/llama-cli
LLAMA_CLI=./build_release/bin/llama-cli
BASE_DIR=/mnt/playground/npu_backend_paper_experiments
MODELS_DIR=$BASE_DIR/models
PROMPTS_DIR=./prompts
# LOG_DIR=$BASE_DIR/LOGS
LOG_DIR=./logs
mkdir -p $LOG_DIR

mode="pipe"
use_cpu="1"

if [ $# -gt 0 ]; then
    mode="$1"
    shift
fi

if [ $# -gt 0 ]; then
    use_cpu="$1"
    shift
fi

echo "mode: $mode"
echo "use_cpu: $use_cpu"

if [ "$mode" = "help" ]; then
    echo "usage: $0 [compile|run|pipe|one|check|parse|clear|help]"
    echo "this script is used to run llama.cpp llama-cli with rknn backend in batch mode"
    echo "npu & cpu 4 threads comparison"
    echo "compile: compile the code"
    echo "run: run the code"
    echo "pipe: compile -> run"
    echo "one: run one model"
    exit 0
fi

if [ "$mode" = "compile" ] || [ "$mode" = "pipe" ] || [ "$mode" = "one" ]; then
    cmake -DGGML_RKNN=ON -DCMAKE_BUILD_TYPE=Release -B build_release -DGGML_RKNN_BACKEND_IMPL=ON
    cmake --build build_release --config Release -j4 --target llama-cli rknn-matmul
    echo "compile done"
fi

if [ "$mode" = "run" ] || [ "$mode" = "pipe" ] || [ "$mode" = "one" ]; then
  for m in  $(ls $MODELS_DIR/*.gguf | sort);
  do
  #   for p in $PROMPTS_DIR/*.txt;
    for p in $PROMPTS_DIR/prompt-100.txt;
    do
      #echo $LOG_DIR/$(basename $m .gguf)_$PREFILL_prefill_$DECODE_decode_$(date +"%m%d-%H%M").log
      echo $m $p NPU
      NPU_PREFILL=1 NPU_DECODE=1 taskset -c 4-7 $LLAMA_CLI -m $m -f $p -n 15 -t 4 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)_npu_$(date +"%m%d-%H%M").log

      if [ "$use_cpu" = "1" ]; then
        echo $m $p CPU
        NPU_PREFILL=0 NPU_DECODE=0 taskset -c 4-7 $LLAMA_CLI -m $m -f $p -n 15 -t 4 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)_cpu_$(date +"%m%d-%H%M").log
      fi
      #echo $LLAMA_CLI -m $m -p \"Hello, how are you?\" -n 10 -t 3 -no-cnv

    if [ "$mode" = "one" ]; then
      exit 0
    fi

    done
  done
fi 

if [ "$mode" = "check" ]; then
    grep -E "Robert" logs/*pu* | cut -d' ' -f1,100-
fi 

if [ "$mode" = "parse" ]; then
    grep -E "eval time" $LOG_DIR/* > ./results/tps_10token.txt
    python ./results/get_tps.py ./results/tps_10token.txt ./results/tps_10token.csv
    echo "parse done"
fi 

if [ "$mode" = "clear" ]; then
    rm -rf $LOG_DIR
    echo "clear done"
fi 

