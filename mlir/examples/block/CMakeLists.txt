add_custom_target(Block)
set_target_properties(Block PROPERTIES FOLDER Examples)

set(LLVM_LINK_COMPONENTS
        Support)

add_subdirectory(include/)
include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)

add_llvm_example(blockc
        blockc.cpp

        mlir/MLIRGen.cpp
        mlir/Dialect.cpp

        parser/CommonParser.cpp
        parser/BitsParser.cpp
        parser/BoolParser.cpp)
add_dependencies(blockc BlockOpsIncGen)

target_link_libraries(blockc
        PRIVATE
        MLIRAnalysis
        MLIRIR
        MLIRParser
        MLIRTransforms)