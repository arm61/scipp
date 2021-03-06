# SPDX-License-Identifier: GPL-3.0-or-later Copyright (c) 2019 Scipp
# contributors (https://github.com/scipp)
set(TARGET_NAME "scipp-dataset")
set(INC_FILES
    ${dataset_INC_FILES}
    include/scipp/dataset/bin.h
    include/scipp/dataset/bins.h
    include/scipp/dataset/choose.h
    include/scipp/dataset/counts.h
    include/scipp/dataset/dataset_access.h
    include/scipp/dataset/dataset.h
    include/scipp/dataset/dataset_util.h
    include/scipp/dataset/except.h
    include/scipp/dataset/groupby.h
    include/scipp/dataset/histogram.h
    include/scipp/dataset/map_view_forward.h
    include/scipp/dataset/map_view.h
    include/scipp/dataset/math.h
    include/scipp/dataset/rebin.h
    include/scipp/dataset/reduction.h
    include/scipp/dataset/special_values.h
    include/scipp/dataset/util.h
    include/scipp/dataset/slice.h
    include/scipp/dataset/shape.h
    include/scipp/dataset/sort.h
    include/scipp/dataset/string.h
)

set(SRC_FILES
    ${dataset_SRC_FILES}
    arithmetic.cpp
    bin.cpp
    bins.cpp
    counts.cpp
    data_array.cpp
    dataset_access.cpp
    dataset.cpp
    except.cpp
    groupby.cpp
    histogram.cpp
    map_view.cpp
    operations.cpp
    rebin.cpp
    reduction.cpp
    util.cpp
    shape.cpp
    slice.cpp
    sort.cpp
    special_values.cpp
    string.cpp
    variable_instantiate_bucket_elements.cpp
    variable_instantiate_dataset.cpp
    variable_reduction.cpp
)

set(LINK_TYPE "STATIC")
if(DYNAMIC_LIB)
  set(LINK_TYPE "SHARED")
endif(DYNAMIC_LIB)

add_library(${TARGET_NAME} ${LINK_TYPE} ${INC_FILES} ${SRC_FILES})
generate_export_header(${TARGET_NAME})
target_link_libraries(${TARGET_NAME} PUBLIC scipp-variable Boost::boost)
if(TBB_FOUND)
  target_link_libraries(${TARGET_NAME} PUBLIC TBB::tbb)
endif()

target_include_directories(
  ${TARGET_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         ${CMAKE_CURRENT_SOURCE_DIR}
         ${CMAKE_CURRENT_BINARY_DIR}
)

set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
add_subdirectory(test)

if(DYNAMIC_LIB)
  install(TARGETS ${TARGET_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif(DYNAMIC_LIB)
