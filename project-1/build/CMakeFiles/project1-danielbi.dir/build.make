# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/build

# Include any dependencies generated for this target.
include CMakeFiles/project1-danielbi.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/project1-danielbi.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/project1-danielbi.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/project1-danielbi.dir/flags.make

CMakeFiles/project1-danielbi.dir/main.cpp.o: CMakeFiles/project1-danielbi.dir/flags.make
CMakeFiles/project1-danielbi.dir/main.cpp.o: /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/main.cpp
CMakeFiles/project1-danielbi.dir/main.cpp.o: CMakeFiles/project1-danielbi.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/project1-danielbi.dir/main.cpp.o"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/project1-danielbi.dir/main.cpp.o -MF CMakeFiles/project1-danielbi.dir/main.cpp.o.d -o CMakeFiles/project1-danielbi.dir/main.cpp.o -c /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/main.cpp

CMakeFiles/project1-danielbi.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/project1-danielbi.dir/main.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/main.cpp > CMakeFiles/project1-danielbi.dir/main.cpp.i

CMakeFiles/project1-danielbi.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/project1-danielbi.dir/main.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/main.cpp -o CMakeFiles/project1-danielbi.dir/main.cpp.s

# Object files for target project1-danielbi
project1__danielbi_OBJECTS = \
"CMakeFiles/project1-danielbi.dir/main.cpp.o"

# External object files for target project1-danielbi
project1__danielbi_EXTERNAL_OBJECTS =

compiled/project1-danielbi: CMakeFiles/project1-danielbi.dir/main.cpp.o
compiled/project1-danielbi: CMakeFiles/project1-danielbi.dir/build.make
compiled/project1-danielbi: CMakeFiles/project1-danielbi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable compiled/project1-danielbi"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/project1-danielbi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/project1-danielbi.dir/build: compiled/project1-danielbi
.PHONY : CMakeFiles/project1-danielbi.dir/build

CMakeFiles/project1-danielbi.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/project1-danielbi.dir/cmake_clean.cmake
.PHONY : CMakeFiles/project1-danielbi.dir/clean

CMakeFiles/project1-danielbi.dir/depend:
	cd /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1 /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1 /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/build /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/build /Users/danielbi/git-repo-no-sync/cs5330-fall23/project-1/build/CMakeFiles/project1-danielbi.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/project1-danielbi.dir/depend
