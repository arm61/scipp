# SPDX-License-Identifier: GPL-3.0-or-later Copyright (c) 2019 Scipp
# contributors (https://github.com/scipp)
set(TARGET_NAME "scipp-variable-test")
add_dependencies(all-tests ${TARGET_NAME})
add_executable(
  ${TARGET_NAME} EXCLUDE_FROM_ALL
  bucket_model_test.cpp
  comparison_test.cpp
  creation_test.cpp
  cumulative_test.cpp
  indexed_slice_view_test.cpp
  mean_test.cpp
  operations_test.cpp
  rebin_test.cpp
  reduce_logical_test.cpp
  reduce_various_test.cpp
  shape_test.cpp
  sort_test.cpp
  subspan_view_test.cpp
  test_variables.cpp
  transform_test.cpp
  trigonometry_test.cpp
  util_test.cpp
  variable_bucket_non_owning_test.cpp
  variable_bucket_test.cpp
  variable_custom_type_test.cpp
  variable_keyword_args_constructor_test.cpp
  variable_math_test.cpp
  variable_scalar_accessors_test.cpp
  variable_special_values_test.cpp
  variable_test.cpp
)
include_directories(SYSTEM ${GMOCK_INCLUDE_DIR} ${GTEST_INCLUDE_DIR})
target_link_libraries(
  ${TARGET_NAME}
  LINK_PRIVATE
  scipp-variable
  gtest_main
  scipp_test_helpers
  ${GTEST_LIBRARIES}
  ${GMOCK_LIBRARIES}
)

if(WIN32)
  target_compile_options(${TARGET_NAME} PRIVATE "/bigobj")
endif()

set_property(
  TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION
                                 ${INTERPROCEDURAL_OPTIMIZATION_TESTS}
)
add_sanitizers(${TARGET_NAME})
if(${WITH_CTEST})
  gtest_discover_tests(${TARGET_NAME} TEST_PREFIX scipp/variable/)
endif()
