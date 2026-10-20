#!/usr/bin/env python3
"""Convert the pinned upstream MI-GAN ONNX checkpoint to canonical safetensors.

Maintainer-only utility. It never downloads code or weights and never overwrites output.
Requires Python, NumPy, and ONNX 1.19.x; the end-user application does not use this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import struct
import sys
import unittest
import uuid
from pathlib import Path
from typing import BinaryIO, Mapping


SOURCE_SHA256 = "593eba0b7e04730f1b61c0a3cbca68d97d8d6a7ff5c6a44a7b9d7fcd880fc5ae"
SOURCE_BYTES = 29_546_882
SOURCE_REVISION = "406830d0fa60666da0071c342ad2fbc8f30c5c64"
EXPECTED_TENSOR_COUNT = 525
EXPECTED_OUTPUT_BYTES = 29_528_324
EXPECTED_OUTPUT_SHA256 = "365c3680e7e33b140761d11a0e62d7ed596572d036bbc3e670540c2baaf9c845"
METADATA = {
    "source_onnx_sha256": SOURCE_SHA256,
    "source_revision": SOURCE_REVISION,
}


class ConversionError(RuntimeError):
    """A provenance, ONNX, tensor, or output error safe to show to a maintainer."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path) -> None:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ConversionError(f"cannot read input ONNX file {path}: {error}") from error
    if size != SOURCE_BYTES:
        raise ConversionError(f"input size is {size} bytes; pinned MI-GAN ONNX must be exactly {SOURCE_BYTES} bytes")
    digest = sha256_file(path)
    if digest != SOURCE_SHA256:
        raise ConversionError(f"input SHA-256 is {digest}; expected pinned MI-GAN SHA-256 {SOURCE_SHA256}")


def numpy_module():
    try:
        import numpy as np
    except ImportError as error:
        raise ConversionError("conversion requires NumPy; see docs/migan-model.md") from error
    return np


def onnx_and_numpy():
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as error:
        raise ConversionError("conversion requires ONNX 1.19.x; see docs/migan-model.md") from error
    version_text = importlib.metadata.version("onnx")
    version = tuple(version_text.split(".")[:2])
    if version != ("1", "19"):
        raise ConversionError(f"expected ONNX 1.19.x for reproducible conversion, found {version_text}")
    return onnx, numpy_helper


def extract_tensors(model, onnx, numpy_helper, np) -> dict[str, object]:
    tensors: dict[str, object] = {}

    def add(name: str, tensor_proto) -> None:
        if not name:
            raise ConversionError("encountered an unnamed ONNX initializer or Constant tensor")
        if tensor_proto.data_location == onnx.TensorProto.EXTERNAL or tensor_proto.external_data:
            raise ConversionError(f"tensor {name!r} uses external ONNX data; expected the pinned self-contained file")
        array = numpy_helper.to_array(tensor_proto)
        if array.dtype not in (np.dtype(np.float32), np.dtype(np.int64)) or array.size == 0:
            return
        if name in tensors:
            raise ConversionError(f"duplicate ONNX tensor name {name!r}")
        tensors[name] = array

    for tensor_proto in model.graph.initializer:
        add(tensor_proto.name, tensor_proto)

    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attribute in node.attribute:
            if attribute.name == "value" and attribute.type == onnx.AttributeProto.TENSOR:
                add(node.output[0], attribute.t)
                break

    if len(tensors) != EXPECTED_TENSOR_COUNT:
        raise ConversionError(f"extracted {len(tensors)} supported nonempty tensors; expected {EXPECTED_TENSOR_COUNT}")
    return tensors


def canonical_header(tensors: Mapping[str, object], np) -> tuple[bytes, list[tuple[str, object]]]:
    ordered = sorted(tensors.items(), key=lambda item: item[0])
    header: dict[str, object] = {"__metadata__": dict(sorted(METADATA.items()))}
    payload_offset = 0
    arrays: list[tuple[str, object]] = []
    for name, value in ordered:
        if not isinstance(name, str) or not name:
            raise ConversionError("safetensors tensor keys must be nonempty strings")
        array = np.asarray(value)
        if array.dtype == np.dtype(np.float32):
            dtype_name, target_dtype = "F32", np.dtype("<f4")
        elif array.dtype == np.dtype(np.int64):
            dtype_name, target_dtype = "I64", np.dtype("<i8")
        else:
            raise ConversionError(f"tensor {name!r} has unsupported dtype {array.dtype}; only F32 and I64 are allowed")
        if array.size == 0:
            raise ConversionError(f"empty tensor {name!r} must not be serialized")
        array = array.astype(target_dtype, copy=False)
        # np.ascontiguousarray promotes zero-dimensional scalar tensors to shape [1].
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        byte_count = int(array.nbytes)
        header[name] = {"dtype": dtype_name, "shape": [int(dim) for dim in array.shape], "data_offsets": [payload_offset, payload_offset + byte_count]}
        payload_offset += byte_count
        arrays.append((name, array))

    # Safetensors reserves this key for metadata. Keep it first, then sort all tensor names;
    # field order and compact UTF-8 JSON are fixed to make output bytes reproducible.
    raw_header = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    padded_header = raw_header + b" " * ((-len(raw_header)) % 8)
    return struct.pack("<Q", len(padded_header)) + padded_header, arrays


def write_safetensors(stream: BinaryIO, tensors: Mapping[str, object], np) -> None:
    header, arrays = canonical_header(tensors, np)
    stream.write(header)
    for _, array in arrays:
        stream.write(array.tobytes(order="C"))


def safe_remove_stage(stage_dir: Path, output_parent: Path, output_name: str) -> None:
    """Remove only this invocation's GUID-named child directory, never a caller path."""
    stage_name = stage_dir.name
    expected = re.fullmatch(re.escape(f".{output_name}.stage-") + r"[0-9a-f]{32}", stage_name)
    resolved_parent = output_parent.resolve(strict=True)
    resolved_stage = stage_dir.resolve(strict=True)
    if not expected or stage_dir.is_symlink() or resolved_stage.parent != resolved_parent:
        raise ConversionError(f"refusing to clean an unexpected staging path: {stage_dir}")
    shutil.rmtree(resolved_stage)


def convert(input_path: Path, output_path: Path) -> tuple[int, str]:
    input_path = input_path.resolve(strict=True)
    output_path = output_path.expanduser().resolve(strict=False)
    if output_path.exists() or output_path.is_symlink():
        raise ConversionError(f"output already exists and will not be overwritten: {output_path}")

    # This verification deliberately precedes all ONNX parsing.
    verify_source(input_path)
    np = numpy_module()
    onnx, numpy_helper = onnx_and_numpy()
    try:
        model = onnx.load_model(str(input_path), load_external_data=False)
    except Exception as error:
        raise ConversionError(f"cannot parse verified MI-GAN ONNX input: {error}") from error
    tensors = extract_tensors(model, onnx, numpy_helper, np)

    output_parent = output_path.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    stage_dir = output_parent / f".{output_path.name}.stage-{uuid.uuid4().hex}"
    try:
        stage_dir.mkdir(exist_ok=False)
    except OSError as error:
        raise ConversionError(f"cannot create unique staging directory {stage_dir}: {error}") from error
    stage_file = stage_dir / output_path.name
    try:
        with stage_file.open("xb") as stream:
            write_safetensors(stream, tensors, np)
            stream.flush()
            os.fsync(stream.fileno())
        size = stage_file.stat().st_size
        digest = sha256_file(stage_file)
        if size != EXPECTED_OUTPUT_BYTES:
            raise ConversionError(f"canonical output is {size} bytes; expected {EXPECTED_OUTPUT_BYTES}")
        if digest != EXPECTED_OUTPUT_SHA256:
            raise ConversionError(f"canonical output digest is {digest}; expected {EXPECTED_OUTPUT_SHA256}")
        # A hard link publishes atomically without replacing a file created by another process.
        try:
            os.link(stage_file, output_path)
        except FileExistsError as error:
            raise ConversionError(f"output appeared during conversion and will not be overwritten: {output_path}") from error
        except OSError as error:
            raise ConversionError(f"cannot atomically publish output on this filesystem: {error}") from error
        return size, digest
    finally:
        if stage_dir.exists():
            safe_remove_stage(stage_dir, output_parent, output_path.name)


class ConverterTests(unittest.TestCase):
    def test_writer_is_deterministic_sorted_and_padded(self) -> None:
        np = numpy_module()
        tensors = {
            "z": np.asarray([2.0, 3.0], dtype=np.float32),
            "a": np.asarray([7], dtype=np.int64),
        }
        import io

        first = io.BytesIO()
        second = io.BytesIO()
        write_safetensors(first, tensors, np)
        write_safetensors(second, tensors, np)
        self.assertEqual(first.getvalue(), second.getvalue())
        header_size = struct.unpack("<Q", first.getvalue()[:8])[0]
        self.assertEqual(header_size % 8, 0)
        header = json.loads(first.getvalue()[8 : 8 + header_size])
        self.assertEqual(list(header), ["__metadata__", "a", "z"])
        self.assertEqual(header["a"]["dtype"], "I64")
        self.assertEqual(header["z"]["dtype"], "F32")

    def test_bad_source_is_rejected_before_onnx_parse_and_output_is_absent(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        input_path = repo_root / "docs" / "licenses" / "MI-GAN-LICENSE.txt"
        output_path = repo_root / "docs" / "licenses" / "must-not-be-created.safetensors"
        with self.assertRaisesRegex(ConversionError, "input size"):
            convert(input_path, output_path)
        self.assertFalse(output_path.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        existing = Path(__file__).resolve().parents[1] / "docs" / "licenses" / "MI-GAN-LICENSE.txt"
        before = sha256_file(existing)
        with self.assertRaisesRegex(ConversionError, "will not be overwritten"):
            convert(existing, existing)
        self.assertEqual(sha256_file(existing), before)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="pinned upstream migan.onnx path")
    parser.add_argument("output", nargs="?", type=Path, help="new safetensors output path (never overwritten)")
    parser.add_argument("--self-test", action="store_true", help="run synthetic canonical-writer and CLI safety tests")
    args = parser.parse_args(argv)
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConverterTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.input is None or args.output is None:
        parser.error("input and output are required unless --self-test is used")
    try:
        size, digest = convert(args.input, args.output)
    except (ConversionError, OSError) as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output} ({size} bytes, sha256={digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
