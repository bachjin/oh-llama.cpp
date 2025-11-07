#!/bin/bash

# Improved deployment script
export TOOLCHAIN_PATH="${HOME}/Downloads/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu"
# export OH_DEVICE_LIB_PATH="/system/lib64"
export OH_DEVICE_LIB_PATH="/system/lib64_glibc"
export OH_DEVICE_BIN_PATH="/data/oh-llama.cpp"
export LLAMA_SOURCE_PATH="${HOME}/oh-llama.cpp"

mode="pipe"

# Parse command line arguments
if [ $# -gt 0 ]; then
    mode="$1"
    shift
fi

echo "MODE: $mode"

if [ "$mode" = "help" ]; then
    echo "usage: $0 [compile|start|get|get-logs|check-logs|lib|clear|pipe]"
    echo "pipe: compile & send, run, get-logs"
    exit 0
fi

if [ "$mode" = "prepare" ]; then
    echo "downloading toolchain to ~/Downloads and cloning oh-llama.cpp to $LLAMA_SOURCE_PATH"

    cd $HOME/Downloads
    wget https://developer.arm.com/-/media/Files/downloads/gnu/11.2-2022.02/binrel/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu.tar.xz
    tar -xzf gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu.tar.xz

    cd $HOME
    git clone https://gitcode.com/openharmony-robot/oh-llama.cpp.git $LLAMA_SOURCE_PATH

    mkdir -p ~/oh-llama.cpp/models
    wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/llama-3.2-1B-Instruct.gguf -O $LLAMA_SOURCE_PATH/models/llama-3.2-1B-Instruct.gguf

    hdc file send $LLAMA_SOURCE_PATH/models/llama-3.2-1B-Instruct.gguf /data/models/Llama-3.2-1B-Instruct__f16.gguf

    echo "TOOLCHAIN_PATH: $TOOLCHAIN_PATH"
    echo "OH_DEVICE_LIB_PATH: $OH_DEVICE_LIB_PATH"
    echo "OH_DEVICE_BIN_PATH: $OH_DEVICE_BIN_PATH"
    echo "LLAMA_SOURCE_PATH: $LLAMA_SOURCE_PATH"
fi

if [ "$mode" = "compile" ] || [ "$mode" = "start" ] || [ "$mode" = "send" ] || [ "$mode" = "pipe" ] || [ "$mode" = "prefill" ]; then
    cd $LLAMA_SOURCE_PATH
    rm -rf $LLAMA_SOURCE_PATH/build 

    # cmake -DCMAKE_BUILD_TYPE=Debug -DGGML_RKNN=ON -DCMAKE_CXX_FLAGS="-fsanitize=address -g -O1" -DCMAKE_BINARY_DIR=${OH_DEVICE_BIN_PATH}/build -DCMAKE_TOOLCHAIN_FILE=hm_aarch64.cmake -DGGML_RKNN_BACKEND_IMPL=ON -B build  \
    

    # cmake -DCMAKE_BUILD_TYPE=Debug -DGGML_RKNN=ON -DCMAKE_CXX_FLAGS="-g3 -O0 -DDEBUG" -DCMAKE_TOOLCHAIN_FILE=hm_aarch64.cmake -DGGML_RKNN_BACKEND_IMPL=ON -B build \
    # && cmake --build build --target llama-cli rknn-matmul rknn-matmul-int --config Debug -j4 \

    cmake -DGGML_RKNN=ON -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=hm_aarch64.cmake -DGGML_RKNN_BACKEND_IMPL=ON -DCMAKE_BINARY_DIR=${OH_DEVICE_BIN_PATH}/build -B build \
    && cmake --build build --target llama-cli rknn-matmul --config Release -j4 \
    || exit 1



    echo "clear & copy application and library files..."

    # hdc shell "rm -rf ${OH_DEVICE_BIN_PATH}/ && mkdir -p ${OH_DEVICE_BIN_PATH}/build ${OH_DEVICE_BIN_PATH}/prompts ${OH_DEVICE_BIN_PATH}/scripts ${OH_DEVICE_BIN_PATH}/logs || echo WHAT??"

    hdc file send $LLAMA_SOURCE_PATH/build ${OH_DEVICE_BIN_PATH}/ 
    hdc file send $LLAMA_SOURCE_PATH/tests_hz/prompts ${OH_DEVICE_BIN_PATH}/
    hdc file send $LLAMA_SOURCE_PATH/scripts/batch-run-oh.sh ${OH_DEVICE_BIN_PATH}/scripts/
    hdc file send $LLAMA_SOURCE_PATH/scripts/batch-run-oh-prefill.sh ${OH_DEVICE_BIN_PATH}/scripts/
    hdc file send $LLAMA_SOURCE_PATH/tests_hz/result_configs/ ${OH_DEVICE_BIN_PATH}/configs/

    hdc shell "chmod +x ${OH_DEVICE_BIN_PATH}/build/bin/llama-cli ${OH_DEVICE_BIN_PATH}/build/bin/rknn-matmul ${OH_DEVICE_BIN_PATH}/build/bin/rknn-matmul-int"
    hdc shell "chmod +x ${OH_DEVICE_BIN_PATH}/scripts/batch-run-oh.sh"

    hdc shell "rm ${OH_DEVICE_BIN_PATH}/build/bin/*.so"

    echo "Copy application libraries to system library directory..."
    for file in $LLAMA_SOURCE_PATH/build/bin/lib*.so; do
        echo "Copying $file"
        chmod +x "$file"
        hdc file send "$file" "${OH_DEVICE_LIB_PATH}/"
        hdc shell "chmod +x '${OH_DEVICE_LIB_PATH}/$(basename \"$file\")'"
    done

    echo "Script execution completed, exiting..."
fi

if [ "$mode" = "run" ] || [ "$mode" = "pipe" ]; then
    echo "Running application..."
    hdc shell "echo performance > /sys/class/devfreq/fdab0000.npu/governor"
    hdc shell "echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor"

    # hdc shell 'export ASAN_OPTIONS="verbosity=1:halt_on_error=1:print_stacktrace=1"'

    # hdc shell 'export LD_LIBRARY_PATH=${OH_DEVICE_LIB_PATH}:/system/lib64:${LD_LIBRARY_PATH}'

    # hdc shell "cd ${OH_DEVICE_BIN_PATH} && LD_LIBRARY_PATH=${OH_DEVICE_LIB_PATH}:/system/lib64:${LD_LIBRARY_PATH} ./scripts/batch-run-oh.sh"

    hdc shell "cd ${OH_DEVICE_BIN_PATH} && ./scripts/batch-run-oh.sh"
fi

if [ "$mode" = "prefill" ]; then
    echo "Running application..."
    hdc shell "echo performance > /sys/class/devfreq/fdab0000.npu/governor"
    hdc shell "echo performance > /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor"

    hdc shell "chmod +x ${OH_DEVICE_BIN_PATH}/scripts/batch-run-oh-prefill.sh"

    hdc shell "cd ${OH_DEVICE_BIN_PATH} && NPU_PREFILL=1 NPU_DECODE=0 ./scripts/batch-run-oh-prefill.sh"
fi

if [ "$mode" = "get" ] || [ "$mode" = "get-logs" ] || [ "$mode" = "pipe" ] || [ "$mode" = "prefill" ]; then
    echo "Getting application logs..."
    hdc file recv ${OH_DEVICE_BIN_PATH}/logs $LLAMA_SOURCE_PATH/
    exit 0
fi

if [ "$mode" = "clear" ]; then
    echo "Clearing application logs..."
    hdc shell "rm -rf ${OH_DEVICE_BIN_PATH}/logs"
    exit 0
fi

if [ "$mode" = "check-logs" ] || [ "$mode" = "check" ]; then
    echo "Listing OH application logs..."
    # hdc shell "ls -l ${OH_DEVICE_BIN_PATH}/logs"
    hdc shell "grep -E Robert ${OH_DEVICE_BIN_PATH}/logs/* --color | cut -d' ' -f1,128-"
fi

if [ "$mode" = "parse-logs" ] || [ "$mode" = "parse" ]; then
    echo "Parsing application logs..."
    grep -E "eval time" $LLAMA_SOURCE_PATH/logs-full-128/* > $LLAMA_SOURCE_PATH/results/tps_10token.txt
    python3 $LLAMA_SOURCE_PATH/results/get_tps.py $LLAMA_SOURCE_PATH/results/tps_10token.txt $LLAMA_SOURCE_PATH/results/tps_10token.csv
    exit 0
fi

if [ "$mode" = "parse-prefill-logs" ] || [ "$mode" = "parse-prefill" ]; then
    echo "Parsing application logs..."
    grep -E "eval time" $LLAMA_SOURCE_PATH/logs/* > $LLAMA_SOURCE_PATH/results/prefill_tps.txt

    python3 $LLAMA_SOURCE_PATH/results/get_tps_prefill.py $LLAMA_SOURCE_PATH/results/prefill_tps.txt $LLAMA_SOURCE_PATH/results/prefill_tps.csv
    exit 0
fi

if [ "$mode" = "lib" ]; then

    echo "Copying system runtime libraries..."

    # hdc shell "rm -rf ${OH_DEVICE_LIB_PATH}"
    hdc shell "mkdir -p ${OH_DEVICE_LIB_PATH}"

    # Copy pthread library
    hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc/lib64/libpthread.so.0" ${OH_DEVICE_LIB_PATH}/

    # Copy dl library
    hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc/lib64/libdl.so.2" ${OH_DEVICE_LIB_PATH}/

    # Copy C++ standard library
    # hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/lib64/libstdc++.so.6" ${OH_DEVICE_LIB_PATH}/
    hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc/usr/lib64/libstdc++.so.6" ${OH_DEVICE_LIB_PATH}/

    # Copy math library
    hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc/lib64/libm.so.6" ${OH_DEVICE_LIB_PATH}/

    # Copy GCC runtime library
    # hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/lib64/libgcc_s.so.1" ${OH_DEVICE_LIB_PATH}/
    hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc/usr/lib64/libgcc_s.so.1" ${OH_DEVICE_LIB_PATH}/

    # Copy C standard library
    hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc/lib64/libc.so.6" ${OH_DEVICE_LIB_PATH}/

    hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/lib64/libasan.so.6" ${OH_DEVICE_LIB_PATH}/
    hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/lib64/libgomp.so.1" ${OH_DEVICE_LIB_PATH}/

    hdc file send $LLAMA_SOURCE_PATH/lib/librknnrt.so ${OH_DEVICE_LIB_PATH}/

    for file in $(ls ${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/lib64/); do 
        echo "copy /$file"
        hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/lib64/$file" ${OH_DEVICE_LIB_PATH}/
    done

    for file in $(ls ${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc/lib64/); do 
        echo "copy $file"
        hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc/lib64/$file" ${OH_DEVICE_LIB_PATH}/
    done

    # Copy dynamic linker - this is critical!
    echo "Copying dynamic linker..."
    hdc shell "mkdir -p /lib"
    hdc file send "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/libc/lib/ld-linux-aarch64.so.1" /lib/

    echo "Setting library path environment variable..."
    hdc shell "export LD_LIBRARY_PATH=${OH_DEVICE_LIB_PATH}:${OH_DEVICE_BIN_PATH}/build/bin:\$LD_LIBRARY_PATH"

    echo "Deployment completed!"
fi

if [ "$mode" = "reboot" ]; then
    echo "reboot..."
    hdc shell reboot
    hdc wait-for-device
    exit 0
fi