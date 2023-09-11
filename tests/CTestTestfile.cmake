# CMake generated Testfile for 
# Source directory: /home/nma5214/.vscodeProjects/clones/llama.cpp/tests
# Build directory: /home/nma5214/.vscodeProjects/clones/llama.cpp/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test-quantize-fns "/home/nma5214/.vscodeProjects/clones/llama.cpp/bin/test-quantize-fns")
set_tests_properties(test-quantize-fns PROPERTIES  _BACKTRACE_TRIPLES "/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;21;add_test;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;25;llama_build_and_test_executable;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;0;")
add_test(test-quantize-perf "/home/nma5214/.vscodeProjects/clones/llama.cpp/bin/test-quantize-perf")
set_tests_properties(test-quantize-perf PROPERTIES  _BACKTRACE_TRIPLES "/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;21;add_test;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;26;llama_build_and_test_executable;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;0;")
add_test(test-sampling "/home/nma5214/.vscodeProjects/clones/llama.cpp/bin/test-sampling")
set_tests_properties(test-sampling PROPERTIES  _BACKTRACE_TRIPLES "/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;21;add_test;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;27;llama_build_and_test_executable;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;0;")
add_test(test-tokenizer-0-llama "/home/nma5214/.vscodeProjects/clones/llama.cpp/bin/test-tokenizer-0-llama" "/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/../models/ggml-vocab-llama.gguf")
set_tests_properties(test-tokenizer-0-llama PROPERTIES  _BACKTRACE_TRIPLES "/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;13;add_test;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;29;llama_test_executable;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;0;")
add_test(test-grammar-parser "/home/nma5214/.vscodeProjects/clones/llama.cpp/bin/test-grammar-parser")
set_tests_properties(test-grammar-parser PROPERTIES  _BACKTRACE_TRIPLES "/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;21;add_test;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;36;llama_build_and_test_executable;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;0;")
add_test(test-llama-grammar "/home/nma5214/.vscodeProjects/clones/llama.cpp/bin/test-llama-grammar")
set_tests_properties(test-llama-grammar PROPERTIES  _BACKTRACE_TRIPLES "/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;21;add_test;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;37;llama_build_and_test_executable;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;0;")
add_test(test-grad0 "/home/nma5214/.vscodeProjects/clones/llama.cpp/bin/test-grad0")
set_tests_properties(test-grad0 PROPERTIES  _BACKTRACE_TRIPLES "/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;21;add_test;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;38;llama_build_and_test_executable;/home/nma5214/.vscodeProjects/clones/llama.cpp/tests/CMakeLists.txt;0;")
