# Add the adapter after upstream has declared its library targets.
set(HULK_BHUMAN_ADAPTER_SOURCE "${CMAKE_CURRENT_LIST_DIR}")
cmake_language(DEFER CALL include "${HULK_BHUMAN_ADAPTER_SOURCE}/CMakeLists.txt")
