cmake_minimum_required(VERSION 3.18)
project(reduce LANGUAGES CUDA CXX)

macro(add_test TEST_NAME TEST_SOURCE)
  set(test_target "${TEST_NAME}_test")
  add_executable(${test_target} ${TEST_SOURCE})
endmacro()


add_test(baseline baseline.cu)
add_test(op1 op1.cu)
add_test(op2 op2.cu)
add_test(op3 op3.cu)
add_test(op4 op4.cu)
add_test(op5 op5.cu)