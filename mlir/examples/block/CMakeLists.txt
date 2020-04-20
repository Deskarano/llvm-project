add_custom_target(Block)
set_target_properties(Block PROPERTIES FOLDER Examples)

add_llvm_example(blockc
        blockc.cpp
        parser/AST.cpp)
include_directories(include/)

target_link_libraries(blockc
        PRIVATE
        MLIRSupport)