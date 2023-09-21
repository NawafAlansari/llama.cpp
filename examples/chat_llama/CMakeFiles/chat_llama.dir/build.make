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
include examples/chat_llama/CMakeFiles/chat_llama.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/chat_llama/CMakeFiles/chat_llama.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/chat_llama/CMakeFiles/chat_llama.dir/progress.make

# Include the compile flags for this target's objects.
include examples/chat_llama/CMakeFiles/chat_llama.dir/flags.make

examples/chat_llama/CMakeFiles/chat_llama.dir/chat_llama.cpp.o: examples/chat_llama/CMakeFiles/chat_llama.dir/flags.make
examples/chat_llama/CMakeFiles/chat_llama.dir/chat_llama.cpp.o: examples/chat_llama/chat_llama.cpp
examples/chat_llama/CMakeFiles/chat_llama.dir/chat_llama.cpp.o: examples/chat_llama/CMakeFiles/chat_llama.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nma5214/.vscodeProjects/clones/llama.cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/chat_llama/CMakeFiles/chat_llama.dir/chat_llama.cpp.o"
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/chat_llama/CMakeFiles/chat_llama.dir/chat_llama.cpp.o -MF CMakeFiles/chat_llama.dir/chat_llama.cpp.o.d -o CMakeFiles/chat_llama.dir/chat_llama.cpp.o -c /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama/chat_llama.cpp

examples/chat_llama/CMakeFiles/chat_llama.dir/chat_llama.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/chat_llama.dir/chat_llama.cpp.i"
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama/chat_llama.cpp > CMakeFiles/chat_llama.dir/chat_llama.cpp.i

examples/chat_llama/CMakeFiles/chat_llama.dir/chat_llama.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/chat_llama.dir/chat_llama.cpp.s"
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama/chat_llama.cpp -o CMakeFiles/chat_llama.dir/chat_llama.cpp.s

# Object files for target chat_llama
chat_llama_OBJECTS = \
"CMakeFiles/chat_llama.dir/chat_llama.cpp.o"

# External object files for target chat_llama
chat_llama_EXTERNAL_OBJECTS = \
"/home/nma5214/.vscodeProjects/clones/llama.cpp/common/CMakeFiles/common.dir/common.cpp.o" \
"/home/nma5214/.vscodeProjects/clones/llama.cpp/common/CMakeFiles/common.dir/console.cpp.o" \
"/home/nma5214/.vscodeProjects/clones/llama.cpp/common/CMakeFiles/common.dir/grammar-parser.cpp.o"

bin/chat_llama: examples/chat_llama/CMakeFiles/chat_llama.dir/chat_llama.cpp.o
bin/chat_llama: common/CMakeFiles/common.dir/common.cpp.o
bin/chat_llama: common/CMakeFiles/common.dir/console.cpp.o
bin/chat_llama: common/CMakeFiles/common.dir/grammar-parser.cpp.o
bin/chat_llama: examples/chat_llama/CMakeFiles/chat_llama.dir/build.make
bin/chat_llama: libllama.a
bin/chat_llama: examples/chat_llama/CMakeFiles/chat_llama.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nma5214/.vscodeProjects/clones/llama.cpp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../bin/chat_llama"
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/chat_llama.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/chat_llama/CMakeFiles/chat_llama.dir/build: bin/chat_llama
.PHONY : examples/chat_llama/CMakeFiles/chat_llama.dir/build

examples/chat_llama/CMakeFiles/chat_llama.dir/clean:
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama && $(CMAKE_COMMAND) -P CMakeFiles/chat_llama.dir/cmake_clean.cmake
.PHONY : examples/chat_llama/CMakeFiles/chat_llama.dir/clean

examples/chat_llama/CMakeFiles/chat_llama.dir/depend:
	cd /home/nma5214/.vscodeProjects/clones/llama.cpp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nma5214/.vscodeProjects/clones/llama.cpp /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama /home/nma5214/.vscodeProjects/clones/llama.cpp /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama /home/nma5214/.vscodeProjects/clones/llama.cpp/examples/chat_llama/CMakeFiles/chat_llama.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/chat_llama/CMakeFiles/chat_llama.dir/depend

