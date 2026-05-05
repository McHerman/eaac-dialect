"""Test bundle JSON serialization for EAAC programs.
"""

import json
import random
import struct
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ElementTypeSpec:
    name: str
    byte_width: int
    random_bytes: Callable[[random.Random, int], bytes]
    decode: Callable[[bytes], list]


ELEMENT_TYPES: dict[str, ElementTypeSpec] = {
    "i8": ElementTypeSpec(
        name="i8",
        byte_width=1,
        random_bytes=lambda rng, n: bytes(rng.randint(0, 255) for _ in range(n)),
        decode=lambda raw: list(raw),
    ),
    # To add e.g. i16: byte_width=2, random_bytes generating 2*n bytes,
    # decode via struct.unpack(f"<{n}h", raw).
}


def make_tensor(shape: list, element_type: str, data: list) -> dict:
    return {"shape": list(shape), "element_type": element_type, "data": list(data)}


def write_bundle(path: str, inputs: list, outputs: list) -> None:
    with open(path, "w") as f:
        json.dump({"inputs": inputs, "outputs": outputs}, f)


def read_bundle(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
