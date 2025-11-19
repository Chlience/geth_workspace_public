# CMake generated Testfile for 
# Source directory: /workspace/gccl/test
# Build directory: /workspace/gccl/build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(AllTestsIngcclUnitTests "/workspace/gccl/build/test/gccl_unit_tests")
set_tests_properties(AllTestsIngcclUnitTests PROPERTIES  WORKING_DIRECTORY "/workspace/gccl/build/test" _BACKTRACE_TRIPLES "/workspace/gccl/test/CMakeLists.txt;15;add_test;/workspace/gccl/test/CMakeLists.txt;0;")
add_test(AllTestsIngcclGPUUnitTests "/workspace/gccl/build/test/gccl_gpu_unit_tests")
set_tests_properties(AllTestsIngcclGPUUnitTests PROPERTIES  WORKING_DIRECTORY "/workspace/gccl/build/test" _BACKTRACE_TRIPLES "/workspace/gccl/test/CMakeLists.txt;30;add_test;/workspace/gccl/test/CMakeLists.txt;0;")
add_test(AllTestsIngcclMPIUnitTests "mpirun" "-np" "3" "--oversubscribe" "--allow-run-as-root" "gccl_mpi_unit_tests")
set_tests_properties(AllTestsIngcclMPIUnitTests PROPERTIES  WORKING_DIRECTORY "/workspace/gccl/build/test" _BACKTRACE_TRIPLES "/workspace/gccl/test/CMakeLists.txt;54;add_test;/workspace/gccl/test/CMakeLists.txt;0;")
