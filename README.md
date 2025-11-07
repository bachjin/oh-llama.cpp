# oh-llama.cpp

## TL;DR

`hdc.sh` contains the script to build and deploy llama.cpp with rknn backend on OpenHarmony device. 

requires: `wget`, `tar`, `git`, `hdc`.

`hdc.sh prepare` to prepare the environment and build the code. Make sure you have enough space in your home directory & Download directory. 

`hdc.sh pipe` to build the code, copy and run the code on the device, logs are saved to `~/oh-llama.cpp/logs`.

## Build

Building llama.cpp with RKNN backend on OpenHarmony. 

Download toolchain: 
```sh
cd ~/Downloads
wget https://developer.arm.com/-/media/Files/downloads/gnu/11.2-2022.02/binrel/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu.tar.xz
tar -xzf gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu.tar.xz
```

Set toolchain path:
```sh
export TOOLCHAIN_PREFIX=~/Downloads/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu
```

Build:
```sh
cmake -DGGML_RKNN=on -DCMAKE_TOOLCHAIN_FILE=hm_aarch64.cmake -DGGML_RKNN_BACKEND_IMPL=ON -B build
cmake --build build --config Release -j4 --target llama-cli
```

## Run
Assume you downloaded this repo to `~/oh-llama.cpp`. 

Download llama-3.2-1B-Instruct.gguf model to OH device: 


You can download the model from Hugging Face:

https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/tree/main

For example, using `wget`:

```sh
mkdir -p ~/oh-llama.cpp/models
wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/llama-3.2-1B-Instruct.gguf -O ~/oh-llama.cpp/models/llama-3.2-1B-Instruct.gguf
```

Push model to OH device:
```sh
hdc file send ~/oh-llama.cpp/models/llama-3.2-1B-Instruct.gguf /data/models/llama-3.2-1B-Instruct.gguf
```

Push libraries to OH device:

```sh
for file in ~/oh-llama.cpp/build/bin/lib*.so; do
    echo "$file"
    hdc file send "$file" /system/lib64/
done
```

Push binaries to OH device:
```sh
hdc file send ~/oh-llama.cpp/build/bin/ /data/llama.cpp/
```

hdc into the device, then run:

```sh
cd /data/oh-llama.cpp/
./build/bin/llama-cli -m /data/models/llama-3.2-1B-Instruct.gguf -p "Hello world!" -n 100 -t 3 -no-cnv
```

