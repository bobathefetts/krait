# cmake -DPROG=<program> [-DARGS=a|b|c] -DEXPECT=<exit code> [-DMATCH=<regex>] -P expect_exit.cmake
# A CTest helper: runs PROG with ARGS ('|'-separated) and passes when it exits with EXPECT and, when MATCH is given, its
# output (stdout + stderr) matches MATCH.
string(REPLACE "|" ";" _args "${ARGS}")
execute_process(COMMAND ${PROG} ${_args} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
message("${out}${err}")
if(NOT "${rc}" STREQUAL "${EXPECT}")
  message(FATAL_ERROR "exit ${rc}, expected ${EXPECT}")
endif()
if(DEFINED MATCH AND NOT "${MATCH}" STREQUAL "" AND NOT "${out}${err}" MATCHES "${MATCH}")
  message(FATAL_ERROR "output does not match '${MATCH}'")
endif()
