#!/usr/bin/env python3
"""Build the Gemma pilot train/dev split from the pinned FRPO math data."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq


SOURCES = {
    "train": (
        "frpo_beta_sweep/20260924/dpao_filter_train_no_math500.parquet",
        "ba63b78e1331f1d7465909c7ac7928861e90013926fe2492e2c530a3bd6dbecb",
        12687,
    ),
    "math500": (
        "math500/test.parquet",
        "f76f2551a7143c93fde82cf353688664f32e4f66b846d44e047bf70542a443ca",
        500,
    ),
    "olympiadbench": (
        "olympiadbench/test.parquet",
        "c66c694dcda9fc7447a32c749c71db6bca0a7d65b79a6b3145b2d1f41d6bcf29",
        674,
    ),
}
INSTRUCTION = r"Let's think step by step and output the final answer within \boxed{}."


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problem_key(prompt):
    question = "\n".join(message["content"] for message in prompt if message["role"] == "user")
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFC", question)).strip()
    problem = normalized.removesuffix(INSTRUCTION).strip()
    if not problem:
        raise ValueError("Missing user problem in math prompt")
    return problem


def write_parquet(path, table):
    if path.exists():
        if not pq.read_table(path).equals(table):
            raise ValueError(f"Existing dataset differs from requested data: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        if not pq.read_table(temporary).equals(table):
            raise ValueError(f"Written data failed verification: {path}")
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing the pinned source Parquets")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "prepared")
    parser.add_argument("--dev-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tables = {}
    sources = {}
    keys = {}
    for name, (relative_path, expected_sha, expected_rows) in SOURCES.items():
        path = args.data_root / relative_path
        source_sha = sha256(path)
        if source_sha != expected_sha:
            raise ValueError(f"Source SHA256 mismatch: {path}")
        table = pq.read_table(path)
        if table.num_rows != expected_rows:
            raise ValueError(f"Source row count mismatch: {path}")
        tables[name] = table
        keys[name] = [problem_key(prompt) for prompt in table.column("prompt").to_pylist()]
        sources[name] = {"path": relative_path, "sha256": source_sha, "rows": expected_rows}

    train = tables["train"]
    train_keys = keys["train"]
    held_out = {name: set(keys[name]) for name in ("math500", "olympiadbench")}
    test_keys = set.union(*held_out.values())
    first_row = {}
    for row_index, key in enumerate(train_keys):
        if key not in test_keys:
            first_row.setdefault(key, row_index)
    if not 0 < args.dev_size < len(first_row):
        raise ValueError("dev-size must leave unique problems for training")

    def dev_rank(key):
        return hashlib.sha256(f"{args.seed}\n{key}".encode()).digest()

    ranked = sorted(first_row, key=dev_rank)
    dev_keys = set(ranked[: args.dev_size])
    train_indices = [i for i, key in enumerate(train_keys) if key not in test_keys and key not in dev_keys]
    tables["train"] = train.take(train_indices)
    tables["dev"] = train.take(sorted(first_row[key] for key in dev_keys))
    tables["smoke_dev"] = train.take(sorted(first_row[key] for key in ranked[: min(32, args.dev_size)]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, table in tables.items():
        path = args.output_dir / f"{name}.parquet"
        write_parquet(path, table)
        outputs[name] = {"filename": path.name, "rows": table.num_rows, "sha256": sha256(path)}

    manifest = {
        "sources": sources,
        "outputs": outputs,
        "dev_seed": args.seed,
        "normalization": "Unicode NFC, whitespace and the shared boxed-answer instruction",
        "overlapping_train_rows": {
            name: sum(key in test for key in train_keys) for name, test in held_out.items()
        },
        "removed_for_dev": sum(key in dev_keys for key in train_keys),
    }
    manifest_json = json.dumps(manifest, indent=2) + "\n"
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and manifest_path.read_text() != manifest_json:
        raise ValueError(f"Existing dataset manifest differs: {manifest_path}")
    manifest_path.write_text(manifest_json)
    print(manifest_json, end="")


if __name__ == "__main__":
    main()
