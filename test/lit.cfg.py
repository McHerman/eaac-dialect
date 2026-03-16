import os

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "EAAC"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.excludes = ["input.mlir", "input_8.mlir", "input_8_small.mlir",
                   "input_8_tiny.mlir", "gemm.mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.eaac_test_build_dir

# Make FileCheck and other LLVM tools available
llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# Set up eaac-opt substitution
tools = [ToolSubst("%eaac-opt", config.eaac_opt)]
llvm_config.add_tool_substitutions(tools, [])
