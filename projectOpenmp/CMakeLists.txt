# 指定cmake最低版本
cmake_minimum_required(VERSION 3.22)

# 项目名称
project(OpenmpDemo)

# 添加编译选项，指定文件编码为UTF-8，add_compile_options需要在add_executable之前
add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

add_executable(OpenmpDemo main.cpp)

FIND_PACKAGE(OpenMP REQUIRED)
if (OPENMP_FOUND)
    message("OPENMP FOUND")
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMp_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMp_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
endif ()

