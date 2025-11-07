cmake_minimum_required(VERSION 3.14)
project(rknn-matmul)

# Get toolchain path from environment or use default
if(DEFINED ENV{TOOLCHAIN_PATH})
    set(TOOLCHAIN_PATH $ENV{TOOLCHAIN_PATH})
else()
    set(TOOLCHAIN_PATH "${HOME}/Downloads/gcc-arm-11.2-2022.02-x86_64-aarch64-none-linux-gnu")
endif()

# Set cross-compilation C++ header paths
set(CROSS_CXX_INCLUDE_PATH "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/include/c++/11.2.1")
set(CROSS_CXX_TARGET_INCLUDE_PATH "${TOOLCHAIN_PATH}/aarch64-none-linux-gnu/include/c++/11.2.1/aarch64-none-linux-gnu")

set(TARGET rknn-matmul)
add_executable(${TARGET} rknn-matmul.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_options(${TARGET} PRIVATE -std=gnu++17)
# Help clangd find libstdc++ headers when using cross compiler
target_compile_options(${TARGET} PRIVATE 
    -isystem ${CROSS_CXX_INCLUDE_PATH}
    -isystem ${CROSS_CXX_TARGET_INCLUDE_PATH}
)

set(TARGET2 rknn-matmul-int)
add_executable(${TARGET2} rknn-matmul-int.cpp)
install(TARGETS ${TARGET2} RUNTIME)
target_link_libraries(${TARGET2} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET2} PRIVATE cxx_std_17)
target_compile_options(${TARGET2} PRIVATE -std=gnu++17)
target_include_directories(${TARGET2} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
# Help clangd find libstdc++ headers when using cross compiler
target_compile_options(${TARGET2} PRIVATE 
    -isystem ${CROSS_CXX_INCLUDE_PATH}
    -isystem ${CROSS_CXX_TARGET_INCLUDE_PATH}
)

set(TARGET3 rknn-matmul-many)
add_executable(${TARGET3} rknn-matmul-many.cpp)
install(TARGETS ${TARGET3} RUNTIME)
target_link_libraries(${TARGET3} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET3} PRIVATE cxx_std_17)
target_compile_options(${TARGET3} PRIVATE -std=gnu++17)
target_include_directories(${TARGET3} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/src)
# Help clangd find libstdc++ headers when using cross compiler
target_compile_options(${TARGET3} PRIVATE 
    -isystem ${CROSS_CXX_INCLUDE_PATH}
    -isystem ${CROSS_CXX_TARGET_INCLUDE_PATH}
)

if (GGML_RKNN)
    add_compile_definitions(GGML_USE_RKNN)
endif()
if (GGML_CPU)
    add_compile_definitions(GGML_USE_CPU)
endif()