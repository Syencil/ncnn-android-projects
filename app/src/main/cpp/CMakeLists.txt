# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

include_directories(${CMAKE_SOURCE_DIR}/include)
set(ALL_SOURCES ${CMAKE_SOURCE_DIR}/source/ncnn.cpp ${CMAKE_SOURCE_DIR}/source/retinaface.cpp ${CMAKE_SOURCE_DIR}/source/yolov5.cpp ${CMAKE_SOURCE_DIR}/source/utils.cpp)
#FILE(GLOB ALL_SOURCES "${CMAKE_SOURCE_DIR}/source/*.cpp")

set(ncnn_lib ${CMAKE_SOURCE_DIR}/../jniLibs/arm64-v8a/libncnn.a)
add_library(ncnn_lib STATIC IMPORTED)
set_target_properties(ncnn_lib PROPERTIES IMPORTED_LOCATION ${ncnn_lib})

set(ANDROID_PLATFORM "android-29")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        ncnn_detection

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp ${ALL_SOURCES})

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

find_library( # Sets the name of the path variable.
        android-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        android )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        ncnn_detection
        # Links the target library to the log library
        # included in the NDK.
        ncnn_lib
        jnigraphics
        ${android-lib}
        ${log-lib})