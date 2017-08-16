# opencv 3.3
# tesseract 4.0
# leptonica 1.74

# suggestion: build tesseract first, then opencv.
# all build from souce on github.


cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 11)

project( ocr4real )

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# OPENCV
find_package( OpenCV REQUIRED )

# tessecract and leptonica
find_package( PkgConfig REQUIRED)
pkg_search_module( TESSERACT REQUIRED tesseract )
pkg_search_module( LEPTONICA REQUIRED lept )


include_directories( ${TESSERACT_INCLUDE_DIRS} )
include_directories( ${LEPTONICA_INCLUDE_DIRS} )
link_directories( ${TESSERACT_LIBRARY_DIRS} )
link_directories( ${LEPTONICA_LIBRARY_DIRS} )


add_executable(
  ocr4real
  main.cpp )

# libs
target_link_libraries(
  ocr4real
  ${OpenCV_LIBS}
  ${TESSERACT_LIBRARIES}
  ${LEPTONICA_LIBRARIES}
  )

