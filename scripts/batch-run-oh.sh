#!/data/local/tmp/bin/bash

BASE_DIR=/data/oh-llama.cpp
LLAMA_CLI=$BASE_DIR/build/bin/llama-cli
MODELS_DIR=/data/models
PROMPTS_DIR=$BASE_DIR/prompts
LOG_DIR=$BASE_DIR/logs
# LOG_DIR=$BASE_DIR/LOGS
mkdir -p $LOG_DIR

export OH_DEVICE_LIB_PATH="/system/lib64_glibc"

export LD_LIBRARY_PATH=${OH_DEVICE_LIB_PATH}:/data/oh-llama.cpp/build/bin:${LD_LIBRARY_PATH}


# ls /data/models/  
model_names=(                                                           
# "Llama-3.2-1B-Instruct__q8_0.gguf"
"Llama-3.2-1B-Instruct__f16.gguf"
# "Llama-3.2-3B-Instruct__q8_0.gguf"
# "Llama-3.2-3B-Instruct__f16.gguf"
# "Phi-4-mini-instruct__f16.gguf"
# "Phi-4-mini-instruct__q8_0.gguf"
# "Qwen2.5-0.5B__f16.gguf"
# "Qwen2.5-0.5B__q8_0.gguf"
# "Qwen2.5-1.5B__f16.gguf"
# "Qwen2.5-1.5B__q8_0.gguf"
# "Qwen2.5-3B__f16.gguf"
# "Qwen2.5-3B__q8_0.gguf"
# "gemma-3-1b-pt__f16.gguf"
# "gemma-3-1b-pt__q8_0.gguf"
# "gemma-3-270m-it__f16.gguf"
# "gemma-3-270m-it__q8_0.gguf"
# "gemma-3-4b-pt__f16.gguf"
# "gemma-3-4b-pt__q8_0.gguf"
)

N_OUTPUT_TOKENS=100
# for m in $MODELS_DIR/*.gguf;
for m_name in ${model_names[*]};
do
  m=$MODELS_DIR/$m_name
  # to warmup
  # NPU_DECODE=1 NPU_PREFILL=1 $LLAMA_CLI -m $m -p "Hello" -n 1 -t 1 -no-cnv --ignore-eos  &> /dev/null
  for p in $PROMPTS_DIR/prompt-128.txt;
  do
    #echo $LOG_DIR/$(basename $m .gguf)_$PREFILL_prefill_$DECODE_decode_$(date +"%m%d-%H%M").log
    # Naming convention of log files: model__prompt__npu/cpu__threads__datetime
    echo $m $p NPU

    model_basename=$(basename $m .gguf)
    # cp ${BASE_DIR}/configs/${model_basename}_128.json ${BASE_DIR}/scripts/rknn-config.json

    export NPU_DECODE=1
    export NPU_PREFILL=1
    # if ! [ -f $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__npu__1.log ]; then
    #   taskset 10 $LLAMA_CLI -m $m -f $p -n $N_OUTPUT_TOKENS -t 1 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__npu__1.log
    #   echo "NPU 1 done"
    # fi
    # if ! [ -f $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__npu__2.log ]; then
    #   taskset 30 $LLAMA_CLI -m $m -f $p -n $N_OUTPUT_TOKENS -t 2 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__npu__2.log
    #   echo "NPU 2 done"
    # fi
    # if ! [ -f $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__npu__3.log ]; then
    #   taskset 70 $LLAMA_CLI -m $m -f $p -n $N_OUTPUT_TOKENS -t 3 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__npu__3.log
    #   echo "NPU 3 done"
    # fi
    if ! [ -f $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__npu__4.log ]; then
      taskset f0 $LLAMA_CLI -m $m -f $p -n $N_OUTPUT_TOKENS -t 4 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__npu__4.log
      echo "NPU 4 done"
    fi

    echo $m $p CPU
    export NPU_DECODE=0
    export NPU_PREFILL=0
    # if ! [ -f $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__cpu__1.log ]; then
    #   taskset 10 $LLAMA_CLI -m $m -f $p -n $N_OUTPUT_TOKENS -t 1 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__cpu__1.log
    #   echo "CPU 1 done"
    # fi
    # if ! [ -f $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__cpu__2.log ]; then
    #   taskset 30 $LLAMA_CLI -m $m -f $p -n $N_OUTPUT_TOKENS -t 2 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__cpu__2.log
    #   echo "CPU 2 done"
    # fi
    # if ! [ -f $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__cpu__3.log ]; then
    #   taskset 70 $LLAMA_CLI -m $m -f $p -n $N_OUTPUT_TOKENS -t 3 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__cpu__3.log
    #   echo "CPU 3 done"
    # fi
    # if ! [ -f $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__cpu__4.log ]; then
    #   taskset f0 $LLAMA_CLI -m $m -f $p -n $N_OUTPUT_TOKENS -t 4 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)__$(basename $p .txt)__cpu__4.log
    #   echo "CPU 4 done"
    # fi

    #echo $LLAMA_CLI -m $m -p \"Hello, how are you?\" -n 10 -t 3 -no-cnv --ignore-eos

    # exit 0
  done
done




# for m in  $(ls $MODELS_DIR/*q8_0.gguf | sort);
# do
#   # for p in $PROMPTS_DIR/prompt-8.txt;
#   for p in $PROMPTS_DIR/prompt-100.txt;
#   do
#     #echo $LOG_DIR/$(basename $m .gguf)_$PREFILL_prefill_$DECODE_decode_$(date +"%m%d-%H%M").log
#     echo $m $p NPU
#     NPU_PREFILL=1 NPU_DECODE=1 taskset 70 $LLAMA_CLI -m $m -f $p -n 100 -t 3 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)_npu_$(date +"%m%d-%H%M").log

#     echo $m $p CPU
#     NPU_PREFILL=0 NPU_DECODE=0 taskset 70 $LLAMA_CLI -m $m -f $p -n 15 -t 4 -no-cnv --ignore-eos &> $LOG_DIR/$(basename $m .gguf)_cpu_$(date +"%m%d-%H%M").log

#     #echo $LLAMA_CLI -m $m -p \"Hello, how are you?\" -n 10 -t 3 -no-cnv

#     exit 0
#   done
# done
