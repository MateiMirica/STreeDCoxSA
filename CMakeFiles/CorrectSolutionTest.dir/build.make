# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

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
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.29.2/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.29.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/mateimirica/Desktop/pystreed

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/mateimirica/Desktop/pystreed

# Include any dependencies generated for this target.
include CMakeFiles/CorrectSolutionTest.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CorrectSolutionTest.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CorrectSolutionTest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CorrectSolutionTest.dir/flags.make

CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.o: CMakeFiles/CorrectSolutionTest.dir/flags.make
CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.o: test/correct_solution_test.cpp
CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.o: CMakeFiles/CorrectSolutionTest.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/mateimirica/Desktop/pystreed/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.o -MF CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.o.d -o CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.o -c /Users/mateimirica/Desktop/pystreed/test/correct_solution_test.cpp

CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/mateimirica/Desktop/pystreed/test/correct_solution_test.cpp > CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.i

CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/mateimirica/Desktop/pystreed/test/correct_solution_test.cpp -o CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.s

# Object files for target CorrectSolutionTest
CorrectSolutionTest_OBJECTS = \
"CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.o"

# External object files for target CorrectSolutionTest
CorrectSolutionTest_EXTERNAL_OBJECTS =

CorrectSolutionTest: CMakeFiles/CorrectSolutionTest.dir/test/correct_solution_test.cpp.o
CorrectSolutionTest: CMakeFiles/CorrectSolutionTest.dir/build.make
CorrectSolutionTest: libSTREEDLib.a
CorrectSolutionTest: CMakeFiles/CorrectSolutionTest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/mateimirica/Desktop/pystreed/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CorrectSolutionTest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CorrectSolutionTest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CorrectSolutionTest.dir/build: CorrectSolutionTest
.PHONY : CMakeFiles/CorrectSolutionTest.dir/build

CMakeFiles/CorrectSolutionTest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CorrectSolutionTest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CorrectSolutionTest.dir/clean

CMakeFiles/CorrectSolutionTest.dir/depend:
	cd /Users/mateimirica/Desktop/pystreed && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/mateimirica/Desktop/pystreed /Users/mateimirica/Desktop/pystreed /Users/mateimirica/Desktop/pystreed /Users/mateimirica/Desktop/pystreed /Users/mateimirica/Desktop/pystreed/CMakeFiles/CorrectSolutionTest.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/CorrectSolutionTest.dir/depend
