set(TARGET refactored-llama-cli)

add_executable(${TARGET}
    refactored_main.cpp
    model_manager.cpp
    session_manager.cpp
    input_handler.cpp
    output_handler.cpp
    config.cpp
    error_handler.cpp
)

target_include_directories(${TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

install(TARGETS ${TARGET} RUNTIME)

target_link_libraries(${TARGET} PRIVATE
    common
    llama
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(${TARGET} PRIVATE cxx_std_11)
