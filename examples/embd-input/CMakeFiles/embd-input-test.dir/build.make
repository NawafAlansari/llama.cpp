# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nma5214/.vscodeProjects/clones/llama.cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nma5214/.vscodeProjects/clones/llama.cpp

# Include any dependencies generated for this target.
include examples/embd-input/CMakeFiles/embd-input-test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/embd-input/CMakeFiles/embd-input-test.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/embd-input/CMakeFiles/embd-input-test.dir/progress.make

# Include the compile flags for this target's objects.
include examples/embd-input/CMakeFiles/embd-input-test.dir/flags.make

examples/embd-input/CMakeFiles/embd-input-test.dir/embd-input-test.cpp.o: examples/embd-input/CMakeFiles/embd-input-test.dir/flags.make
examples/embd-input/CMakeFiles/embd-input-test.dir/embd-input-test.cpp.o: examples/embd-input/embd-input-test.cpp
examples/embd-input/CMakeFiles/embd-input-test.dir/embd-input-test.cpp.o: examples/embd-input/CMakeFiles/embd-input-test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nma5214/.vscodeProjects/clones/llama.cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/embd-input/CMakeFiles/embd-input-test.dir/embd-input-test.cpp.o"
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/embd-input/CMakeFiles/embd-input-test.dir/embd-input-test.cpp.o -MF CMakeFiles/embd-input-test.dir/embd-input-test.cpp.o.d -o CMakeFiles/embd-input-test.dir/embd-input-test.cpp.o -c /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input/embd-input-test.cpp

examples/embd-input/CMakeFiles/embd-input-test.dir/embd-input-test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/embd-input-test.dir/embd-input-test.cpp.i"
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input/embd-input-test.cpp > CMakeFiles/embd-input-test.dir/embd-input-test.cpp.i

examples/embd-input/CMakeFiles/embd-input-test.dir/embd-input-test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/embd-input-test.dir/embd-input-test.cpp.s"
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input/embd-input-test.cpp -o CMakeFiles/embd-input-test.dir/embd-input-test.cpp.s

# Object files for target embd-input-test
embd__input__test_OBJECTS = \
"CMakeFiles/embd-input-test.dir/embd-input-test.cpp.o"

# External object files for target embd-input-test
embd__input__test_EXTERNAL_OBJECTS = \
"/home/nma5214/.vscodeProjects/clones/llama.cpp/common/CMakeFiles/common.dir/common.cpp.o" \
"/home/nma5214/.vscodeProjects/clones/llama.cpp/common/CMakeFiles/common.dir/console.cpp.o" \
"/home/nma5214/.vscodeProjects/clones/llama.cpp/common/CMakeFiles/common.dir/grammar-parser.cpp.o"

bin/embd-input-test: examples/embd-input/CMakeFiles/embd-input-test.dir/embd-input-test.cpp.o
bin/embd-input-test: common/CMakeFiles/common.dir/common.cpp.o
bin/embd-input-test: common/CMakeFiles/common.dir/console.cpp.o
bin/embd-input-test: common/CMakeFiles/common.dir/grammar-parser.cpp.o
bin/embd-input-test: examples/embd-input/CMakeFiles/embd-input-test.dir/build.make
bin/embd-input-test: libllama.a
bin/embd-input-test: examples/embd-input/libembdinput.a
bin/embd-input-test: libllama.a
bin/embd-input-test: examples/embd-input/CMakeFiles/embd-input-test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nma5214/.vscodeProjects/clones/llama.cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/embd-input-test"
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/embd-input-test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/embd-input/CMakeFiles/embd-input-test.dir/build: bin/embd-input-test
.PHONY : examples/embd-input/CMakeFiles/embd-input-test.dir/build

examples/embd-input/CMakeFiles/embd-input-test.dir/clean:
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input && $(CMAKE_COMMAND) -P CMakeFiles/embd-input-test.dir/cmake_clean.cmake
.PHONY : examples/embd-input/CMakeFiles/embd-input-test.dir/clean

examples/embd-input/CMakeFiles/embd-input-test.dir/depend:
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nma5214/.vscodeProjects/clones/llama.cpp /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input /home/nma5214/.vscodeProjects/clones/llama.cpp /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/embd-input/CMakeFiles/embd-input-test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/embd-input/CMakeFiles/embd-input-test.dir/depend

