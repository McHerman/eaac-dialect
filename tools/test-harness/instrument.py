"""In-Python MLIR rewrite for reference execution.
"""

import os
import random
import sys

import numpy as np

LLVM_BUILD = os.environ.get(
    "LLVM_BUILD", os.path.expanduser("~/dtu/Thesis/MLIR/llvm-project/build")
)
sys.path.insert(0, os.path.join(LLVM_BUILD, "tools/mlir/python_packages/mlir_core"))

from mlir import ir
from mlir.dialects import arith, func, tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundle


def _is_i8_tensor(t) -> bool:
    if not ir.RankedTensorType.isinstance(t):
        return False
    elem = ir.RankedTensorType(t).element_type
    return ir.IntegerType.isinstance(elem) and ir.IntegerType(elem).width == 8


def _replace_args_with_constants(main_op, rng: random.Random) -> list:
    body = main_op.body.blocks[0]
    args = list(body.arguments)
    inputs = []

    with ir.InsertionPoint.at_block_begin(body):
        replacements = []
        for arg in args:
            if not _is_i8_tensor(arg.type):
                raise NotImplementedError(
                    f"random input generation only supports i8 tensors, got {arg.type}"
                )
            rtt = ir.RankedTensorType(arg.type)
            shape = list(rtt.shape)
            n = 1
            for d in shape:
                n *= d
            raw = bytes(rng.randint(0, 255) for _ in range(n))
            arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int8).reshape(shape)
            attr = ir.DenseElementsAttr.get(arr, type=arg.type)
            const = arith.ConstantOp(arg.type, attr)
            replacements.append((arg, const.result))
            inputs.append(bundle.make_tensor(shape, "i8", list(raw)))

    for old, new in replacements:
        old.replace_all_uses_with(new)
    for i in reversed(range(len(args))):
        body.erase_argument(i)

    old_ft = ir.FunctionType(main_op.function_type.value)
    main_op.attributes["function_type"] = ir.TypeAttr.get(
        ir.FunctionType.get(inputs=[], results=old_ft.results)
    )
    return inputs


def _ensure_print_decl(module: ir.Module, unranked_i32) -> None:
    for op in module.body.operations:
        if op.OPERATION_NAME == "func.func" and op.sym_name.value == "printMemrefI32":
            return
    with ir.InsertionPoint.at_block_begin(module.body):
        decl = func.FuncOp(
            "printMemrefI32",
            ir.FunctionType.get(inputs=[unranked_i32], results=[]),
            visibility="private",
        )
        decl.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()


def _add_prints(module: ir.Module, main_op) -> None:
    body = main_op.body.blocks[0]
    i32 = ir.IntegerType.get_signless(32)
    unranked_i32 = ir.UnrankedTensorType.get(i32)
    _ensure_print_decl(module, unranked_i32)

    captured = []
    """
    for op in body.operations:
        if op.OPERATION_NAME == "func.call":
            for r in op.results:
                if _is_i8_tensor(r.type):
                    captured.append(r)
    """

    for op in body.operations:
        #print(op.OPERATION_NAME)

        if op.OPERATION_NAME == "func.return":
            for r in op.operands:
                print(r.type)
                if _is_i8_tensor(r.type):
                    captured.append(r)


    terminator = list(body.operations)[-1]
    with ir.InsertionPoint(terminator):
        for v in captured:
            shape = ir.RankedTensorType(v.type).shape
            ext_t = ir.RankedTensorType.get(shape, i32)
            ext = arith.ExtUIOp(ext_t, v)
            cast = tensor.CastOp(unranked_i32, ext.result)
            func.CallOp([], "printMemrefI32", [cast.result])
        func.ReturnOp([])
    terminator.erase()

    main_op.attributes["function_type"] = ir.TypeAttr.get(
        ir.FunctionType.get(inputs=[], results=[])
    )


def instrument(src: str, seed: int) -> tuple:
    """Rewrite MLIR source for reference execution.

    Returns (rewritten_text, [input_tensor_dict, ...]).
    """
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(src)
        # The DLTI spec is EAAC-only configuration; drop it so the reference
        # path (which feeds mlir-runner) doesn't need DLTI registered.
        module_attrs = module.operation.attributes
        if "dlti.target_system_spec" in module_attrs:
            del module_attrs["dlti.target_system_spec"]
        main_op = None
        for op in module.body.operations:
            if op.OPERATION_NAME == "func.func" and op.sym_name.value == "main":
                main_op = op
                break
        if main_op is None:
            raise ValueError("@main not found in module")

        rng = random.Random(seed)
        inputs = _replace_args_with_constants(main_op, rng)
        _add_prints(module, main_op)
        return str(module), inputs
