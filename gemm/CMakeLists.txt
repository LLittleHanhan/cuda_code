cmake_minimum_required(VERSION 3.18)
project(reduce LANGUAGES CUDA CXX)

macro(add_test TEST_NAME TEST_SOURCE)
  set(test_target "${TEST_NAME}_test")
  add_executable(${test_target} ${TEST_SOURCE})
endmacro()


add_test(baseline baseline.cu)
add_test(op1 op1.cu)
add_test(others others.cu)
add_test(cublas cublas.cu)

execute_process(COMMAND ncu --set full --export ../report/baseline -f baseline_test)
execute_process(COMMAND ncu --set full --export ../report/op1 -f op1_test)