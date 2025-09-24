
configure_file(
    ../../include/model_related_config.h.in
    ${CMAKE_BINARY_DIR}/model_related_config.h
    @ONLY
)
include_directories(${CMAKE_BINARY_DIR})

# include(FetchContent)
# FetchContent_Declare(
#     json
#     GIT_REPOSITORY https://github.com/nlohmann/json.git
#     GIT_TAG v3.11.2
# )
# FetchContent_MakeAvailable(json)

set(TARGET_NAME ggml-rknn)

ggml_add_backend_library(${TARGET_NAME}
                            ggml-rknn.cpp
                            ../../include/ggml-cpu.h
                            ../ggml-cpu/ggml-cpu-impl.h
                            ../../include/ggml-rknn.h)

# set(RKNN_API_PATH /home/hanzhang/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api)

# if(DEFINED SINGLE_THREAD)
#     add_compile_definitions(SINGLE_THREAD=$(SINGLE_THREAD))
# endif()
if(DEFINED GGML_DEBUGING)
    add_compile_definitions(GGML_RKNN_DEBUGING)
endif()



message(STATUS "using rknn")
target_link_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${TARGET_NAME} PUBLIC ggml-cpu ${CMAKE_SOURCE_DIR}/lib/librkllmrt.so ${CMAKE_SOURCE_DIR}/lib/librknnrt.so)
# target_link_libraries(${TARGET_NAME} PUBLIC ggml-cpu ${CMAKE_SOURCE_DIR}/lib/librkllmrt.so ${CMAKE_SOURCE_DIR}/lib/librknnrt.so nlohmann_json::nlohmann_json)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/lib)


# find_library(RKLLMRT NAMES rkllmrt librkllmrt PATHS ${PROJECT_SOURCE_DIR}/lib)

# add_library(ggml-rknn SHARED ggml-rknn.cpp)

# target_include_directories(ggml-rknn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# link_directories($(PROJECT_SOURCE_DIR)/lib)





# target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCL_LIBRARIES})
# target_include_directories(${TARGET_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
