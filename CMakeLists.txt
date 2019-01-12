cmake_minimum_required(VERSION 3.13)

project(CNNObjectDetector)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CNNOD_SRC "src/CNNObjectDetector/")
set(FaceDetector_SRC "src/FaceDetector/")
add_definitions(-DSOURCE="../${CNNOD_SRC}")


option(WITH_SSE "" OFF)
option(WITH_AVX "" ON)
option(WITH_AVX2 "" OFF)
option(WITH_OpenMP "" ON)
option(WITH_CUDA "" OFF)
option(WITH_OpenCL "" OFF)
option(WITH_DETECTOR_PROFILER "" OFF)
option(WITH_CNN_PROFILER "" OFF)
option(WITH_TESTS "" OFF)
option(CHECK_CNN_SSE "" OFF)
option(BUILD_FaceDetector "" OFF)
option(BUILD_FDDBTest "" OFF)


if(WITH_SSE)
  add_definitions(-DUSE_SSE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:SSE2")
else()
  add_definitions(-DUSE_CNTK_MODELS)
endif()
if(WITH_AVX)
  add_definitions(-DUSE_AVX)
  remove_definitions(-DUSE_SSE)
  add_definitions(-DUSE_CNTK_MODELS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
endif()
if(WITH_AVX2)
  add_definitions(-DUSE_AVX2)
  remove_definitions(-DUSE_AVX)
  remove_definitions(-DUSE_SSE)
  add_definitions(-DUSE_CNTK_MODELS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
endif()


if(WITH_DETECTOR_PROFILER)
  add_definitions(-DPROFILE_DETECTOR)
endif()
if(WITH_CNN_PROFILER)
  add_definitions(-DPROFILE_CNN_SIMD)
  add_definitions(-DPROFILE_CNN_CUDA)
  add_definitions(-DPROFILE_CNN_CL)
endif()


if(WITH_OpenMP)
  add_definitions(-DUSE_OMP)
  add_definitions(-DMAX_NUM_THREADS=4)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
  
  find_package(OpenMP)
  if (OPENMP_FOUND)
      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  endif()
endif()


set(SOURCE "${CNNOD_SRC}/config.h" "${CNNOD_SRC}/test_accuracy.cpp")
if(WITH_TESTS)
  set(SOURCE ${SOURCE} "${CNNOD_SRC}/test_perfomance.cpp" "${CNNOD_SRC}/test_conv.cpp" "${CNNOD_SRC}/test_landmark.cpp")
endif()
source_group("" FILES ${SOURCE})


file(GLOB SIMD_SRC
  "${CNNOD_SRC}/*simd*"
  "${CNNOD_SRC}/*cplusplus*"
  "${CNNOD_SRC}/image*"
  "${CNNOD_SRC}/timer*"
)
set(SOURCE ${SOURCE} ${SIMD_SRC})
source_group("SIMD" FILES ${SIMD_SRC})

file(GLOB DET_SRC
  "${CNNOD_SRC}/cnn_detector*"
  "${CNNOD_SRC}/packing_2D*"
)
set(SOURCE ${SOURCE} ${DET_SRC})
source_group("Detector" FILES ${DET_SRC})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set (CMAKE_CXX_STANDARD 11)


if(WITH_CUDA)
   file(GLOB CUDA_SRC
    "${CNNOD_SRC}/*cuda*"
  )
  set(SOURCE ${SOURCE} ${CUDA_SRC})
  
  find_package(CUDA 9.2 REQUIRED)
  enable_language(CUDA)
  
  source_group("CUDA" FILES ${CUDA_SRC})
  add_definitions(-DUSE_CUDA)
endif()


if(WITH_OpenCL)
  file(GLOB CL_SRC
    "${CNNOD_SRC}/*cl*"
  )
  set(SOURCE ${SOURCE} ${CL_SRC})
  
  find_package(OpenCL REQUIRED)
  include_directories(${OpenCL_INCLUDE_DIRS})
  link_directories(${OpenCL_LIBRARY})
  
  source_group("OpenCL" FILES ${CL_SRC})
  add_definitions(-DUSE_CL)
  remove_definitions(-DUSE_CUDA)
endif()


if(CHECK_CNN_SSE)
  file(GLOB LEGACY_SRC
    "${CNNOD_SRC}/cnn.*"
    "${CNNOD_SRC}/af*"
  )
  set(SOURCE ${SOURCE} ${LEGACY_SRC})

  source_group("Legacy" FILES ${LEGACY_SRC})
  add_definitions(-DCHECK_TEST)

  set(ModelsPath "../models/" CACHE STRING "")
  add_definitions(-DMODELSPATH="${ModelsPath}")
  
  add_definitions(-DUSE_SSE)
  remove_definitions(-DUSE_AVX)
  remove_definitions(-DUSE_AVX2)
  remove_definitions(-DUSE_OMP)
  remove_definitions(-DUSE_CUDA)
  remove_definitions(-DUSE_CL)
  remove_definitions(-DUSE_CNTK_MODELS)
endif()


if(BUILD_FaceDetector OR BUILD_FDDBTest)
  file(GLOB API_SRC
    "${CNNOD_SRC}/*API*"
    "${CNNOD_SRC}/cnn_models_converter.*"
  )
  set(SOURCE ${SOURCE} ${API_SRC})
  source_group("API" FILES ${API_SRC})

  add_library(CNNObjectDetector STATIC ${SOURCE})
  add_definitions(-DCNNOBJECTDETECTOR_EXPORTS)
else()
  add_executable(CNNObjectDetector ${SOURCE})
endif()


if(WITH_CUDA)
  target_link_libraries(CNNObjectDetector ${CUDA_LIBRARIES})
endif()


if(WITH_OpenCL)
  target_link_libraries(CNNObjectDetector ${OpenCL_LIBRARY})
endif()


function(build_test proj_name source)
  add_executable(${proj_name} ${source})
  target_link_libraries(${proj_name} CNNObjectDetector)
  include_directories(${proj_name} ${CNNOD_SRC})

  find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc videoio)
  target_link_libraries(${proj_name} ${OpenCV_LIBRARIES})
  include_directories(${proj_name} ${OpenCV_INCLUDE_DIRS})

  set_target_properties(CNNObjectDetector PROPERTIES IMPORTED GLOBAL)
endfunction(build_test)

if(BUILD_FaceDetector)
  build_test(FaceDetector "${FaceDetector_SRC}/main.cpp")
endif()


if(BUILD_FDDBTest)
  set(FDDBPath "G:/QA/fddb/" CACHE STRING "")
  add_definitions(-DFDDBFold="../${FaceDetector_SRC}")
  add_definitions(-DFDDBPath="${FDDBPath}")
  build_test(FDDBTest "${FaceDetector_SRC}/FDDB_test.cpp")
endif()
