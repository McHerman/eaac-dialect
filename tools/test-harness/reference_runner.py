#!/usr/bin/env python3
"""Reference execution of EAAC MLIR programs via mlir-runner.
"""

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bundle
import instrument

SEED = 42

LLVM_BUILD = os.environ.get(
    "LLVM_BUILD", os.path.expanduser("~/dtu/Thesis/MLIR/llvm-project/build")
)
MLIR_OPT = os.path.join(LLVM_BUILD, "bin", "mlir-opt")
MLIR_RUNNER = os.path.join(LLVM_BUILD, "bin", "mlir-runner")
RUNNER_UTILS = os.path.join(LLVM_BUILD, "lib", "libmlir_runner_utils.so")
C_RUNNER_UTILS = os.path.join(LLVM_BUILD, "lib", "libmlir_c_runner_utils.so")


def lower_and_run(src: str) -> str:
    """Lower instrumented MLIR and execute. Returns mlir-runner stdout (text)."""
    bufferize_cmd = [
        MLIR_OPT,
        "--convert-elementwise-to-linalg",
        "--one-shot-bufferize=bufferize-function-boundaries",
        "--canonicalize",
        "--buffer-deallocation-pipeline",
    ]
    buf = subprocess.run(bufferize_cmd, input=src, capture_output=True, text=True)
    if buf.returncode != 0:
        sys.exit(f"mlir-opt (bufferize) failed:\n{buf.stderr}")

    lower_cmd = [
        MLIR_OPT,
        "--convert-linalg-to-loops",
        "--convert-bufferization-to-memref",
        "--expand-strided-metadata",
        "--lower-affine",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--finalize-memref-to-llvm",
        "--convert-func-to-llvm",
        "--reconcile-unrealized-casts",
    ]
    low = subprocess.run(lower_cmd, input=buf.stdout, capture_output=True, text=True)
    if low.returncode != 0:
        sys.exit(f"mlir-opt (lower) failed:\n{low.stderr}")

    runner_cmd = [
        MLIR_RUNNER,
        "-e", "main",
        "--entry-point-result=void",
        f"--shared-libs={C_RUNNER_UTILS},{RUNNER_UTILS}",
    ]
    run = subprocess.run(runner_cmd, input=low.stdout, capture_output=True, text=True)
    if run.returncode != 0:
        sys.exit(f"mlir-runner failed:\n{run.stderr}")
    return run.stdout


def parse_print_output(text: str) -> list:
    """Parse printMemrefI32 stdout into a list of 2D matrices.

    Each block looks like:
        Unranked Memref ... sizes = [R, C] ... data =
        [[v, v, ...],
         [v, v, ...]]
    """
    matrices = []
    for chunk in text.split("Unranked Memref")[1:]:
        m = re.search(r'sizes = \[(\d+), (\d+)\]', chunk)
        if not m:
            continue
        rows, cols = int(m.group(1)), int(m.group(2))
        data = re.search(r'\[\[(.*?)\]\]', chunk, re.DOTALL)
        if not data:
            continue
        values = [int(v) for v in re.findall(r'-?\d+', data.group(1))]
        if len(values) != rows * cols:
            print(f"Warning: expected {rows*cols} values, got {len(values)}",
                  file=sys.stderr)
            continue
        matrices.append([values[i*cols:(i+1)*cols] for i in range(rows)])
    return matrices


def run(src: str, seed: int = SEED) -> tuple:
    """Instrument, lower, execute, and parse.

    Returns (inputs, output_matrices, output_element_widths).
    """
    rewritten, inputs, output_widths = instrument.instrument(src, seed)
    return inputs, parse_print_output(lower_and_run(rewritten)), output_widths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="MLIR source file to execute")
    ap.add_argument(
        "-o", "--output",
        help="Output JSON path (default: <input>.reference.json)",
    )
    args = ap.parse_args()

    with open(args.input) as f:
        src = f.read()

    print("Running instrument → bufferize → lower → execute...", flush=True)
    inputs, matrices, output_widths = run(src)
    print(f"Generated {len(inputs)} input tensors, parsed {len(matrices)} result matrices")

    if not matrices:
        sys.exit("No result matrices parsed — nothing to write.")

    outputs = [
        bundle.make_tensor(
            shape=[len(m), len(m[0])],
            element_type=f"i{width}",
            data=[v for row in m for v in row],
        )
        for m, width in zip(matrices, output_widths)
    ]

    out_path = args.output or os.path.splitext(args.input)[0] + ".reference.json"
    bundle.write_bundle(out_path, inputs=inputs, outputs=outputs)
    print(f"Reference bundle written to {out_path}")

    for idx, mat in enumerate(matrices):
        print(f"\nResult {idx} (top-left 4x4):")
        for row in mat[:4]:
            print(f"  {row[:4]}")


if __name__ == "__main__":
    main()
