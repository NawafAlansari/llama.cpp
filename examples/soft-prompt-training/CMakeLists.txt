# specify the CMake version
cmake_minimum_required(VERSION 3.10)
project(soft-prompt-training)


# --- Dependencies--------------------------------------------------------------


set(TARGET soft_prompt_training)
set(SOURCE_FILES soft_prompt_training.cpp)

# Create executable
add_executable(${TARGET} ${SOURCE_FILES})

# Link libraries
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})


# Specify C++ standard
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# Add any additional dependencies
if(TARGET BUILD_INFO)
  add_dependencies(${TARGET} BUILD_INFO)
endif()

# Install executable
install(TARGETS ${TARGET} RUNTIME)
