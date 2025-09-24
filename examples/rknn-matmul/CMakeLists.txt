cmake_minimum_required(VERSION 3.14)
project(rknn-matmul)

set(TARGET rknn-matmul)
add_executable(${TARGET} rknn-matmul.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_options(${TARGET} PRIVATE -std=gnu++17)
# Help clangd find libstdc++ headers when using clang as driver
target_compile_options(${TARGET} PRIVATE 
    -isystem /usr/include/c++/11
    -isystem /usr/include/aarch64-linux-gnu/c++/11
)
if (GGML_RKNN)
    add_compile_definitions(GGML_USE_RKNN)
endif()
if (GGML_CPU)
    add_compile_definitions(GGML_USE_CPU)
endif()