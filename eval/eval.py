#!/usr/bin/env python3
"""Evaluate math datasets with vLLM sampling and math-verify scoring."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = REPO_ROOT / "verl"
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.utils.reward_score import math_verify  # noqa: E402
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed  # noqa: E402

DEFAULT_DATA_ROOT = "~/rl-kl/verl/data"
DEFAULT_DATASETS = {
    "aime24": f"{DEFAULT_DATA_ROOT}/aime24/test.parquet",
    "aime25": f"{DEFAULT_DATA_ROOT}/aime25/test.parquet",
    "olympiadbench": f"{DEFAULT_DATA_ROOT}/olympiadbench/test.parquet",
    "amc23": f"{DEFAULT_DATA_ROOT}/amc23/test.parquet",
    "minerva_math": f"{DEFAULT_DATA_ROOT}/minerva_math/test.parquet",
}
DEFAULT_OUTPUT_PATH = REPO_ROOT / "eval" / "vllm_math_results.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval" / "results"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str


def env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw_value!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vLLM inference on math parquet datasets and report pass@n plus mean@n accuracy."
    )
    parser.add_argument("--model", required=True, help="HF model name or local model/checkpoint path.")
    parser.add_argument(
        "--data",
        action="append",
        default=None,
        help=(
            "Dataset spec. Repeat for multiple datasets. Supports aliases aime24/aime25, "
            "name=/path/to/test.parquet, or a raw parquet path. Defaults to aime24."
        ),
    )
    parser.add_argument("--output", default=None, help="Explicit JSONL file for rollouts and judgments.")
    parser.add_argument(
        "--output-name",
        default=None,
        help="Output basename under --output-dir. Adds .jsonl when no suffix is provided.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory used by --output-name.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON file for aggregate metrics. Defaults to <output>.summary.json.",
    )
    parser.add_argument("--n", type=int, default=16, help="Number of rollouts per question for pass@n/mean@n.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Sampling top-p.")
    parser.add_argument("--max-response-length", type=int, default=8192, help="Maximum generated tokens.")
    parser.add_argument("--max-model-len", type=int, default=10240, help="vLLM max model length.")
    parser.add_argument(
        "--tensor-parallel-size",
        "--tp",
        dest="tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size.",
    )
    parser.add_argument(
        "--data-parallel-size",
        "--dp",
        dest="data_parallel_size",
        type=int,
        default=1,
        help="vLLM data parallel size.",
    )
    parser.add_argument("--dtype", default="auto", help="vLLM dtype, for example auto, bfloat16, float16.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization.")
    parser.add_argument("--max-num-seqs", type=int, default=64, help="vLLM max_num_seqs.")
    parser.add_argument(
        "--generate-batch-size",
        type=int,
        default=0,
        help=(
            "Number of questions per llm.generate call per local worker. "
            "0 means generate each worker's full shard at once."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=env_int("HOST_NUM", 1),
        help="Number of multi-node data shards. Defaults to HOST_NUM when launched by jobs/run.sh.",
    )
    parser.add_argument(
        "--shard-rank",
        type=int,
        default=env_int("INDEX", 0),
        help="This node's shard rank. Defaults to INDEX when launched by jobs/run.sh.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="When num-shards > 1, rank 0 waits for all shard outputs and writes the final merged output.",
    )
    parser.add_argument(
        "--merge-timeout",
        type=float,
        default=0.0,
        help="Seconds rank 0 waits for shard outputs before failing. 0 means wait forever.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Pass enforce_eager=True to vLLM to avoid cudagraph warmup/capture issues.",
    )
    parser.add_argument(
        "--disable-custom-all-reduce",
        action="store_true",
        help="Pass disable_custom_all_reduce=True to vLLM; useful for flashinfer fused allreduce failures.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for vLLM sampling.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to vLLM.")
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Use raw user content instead of tokenizer.apply_chat_template.",
    )
    parser.add_argument("--verify-timeout", type=float, default=30.0, help="Timeout per math-verify call.")
    return parser.parse_args()


def parse_dataset_spec(spec: str) -> DatasetSpec:
    if spec in DEFAULT_DATASETS:
        return DatasetSpec(name=spec, path=DEFAULT_DATASETS[spec])

    if "=" in spec:
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Invalid dataset spec: {spec!r}")
        return DatasetSpec(name=name, path=path)

    path = Path(spec)
    if path.suffix == ".parquet" or "/" in spec:
        parent_name = path.parent.name
        name = parent_name if parent_name else path.stem
        return DatasetSpec(name=name, path=spec)

    known = ", ".join(sorted(DEFAULT_DATASETS))
    raise ValueError(f"Unknown dataset alias {spec!r}. Known aliases: {known}")


def get_dataset_specs(raw_specs: list[str] | None) -> list[DatasetSpec]:
    specs = raw_specs or ["aime24"]
    dataset_specs = [parse_dataset_spec(spec) for spec in specs]
    seen_names: set[str] = set()
    for dataset_spec in dataset_specs:
        if dataset_spec.name in seen_names:
            raise ValueError(f"Duplicate dataset name: {dataset_spec.name}")
        seen_names.add(dataset_spec.name)
    return dataset_specs


def get_output_path(args: argparse.Namespace) -> Path:
    if args.output and args.output_name:
        raise ValueError("Use either --output or --output-name, not both.")

    if args.output:
        return Path(args.output)

    if args.output_name:
        output_name = Path(args.output_name)
        if output_name.is_absolute() or output_name.parent != Path("."):
            raise ValueError("--output-name must be a basename, not a path. Use --output for explicit paths.")
        if output_name.suffix == "":
            output_name = output_name.with_suffix(".jsonl")
        return Path(args.output_dir) / output_name

    return DEFAULT_OUTPUT_PATH


def get_shard_path(path: Path, shard_rank: int, num_shards: int) -> Path:
    return path.with_name(f"{path.name}.part-{shard_rank:05d}-of-{num_shards:05d}")


def get_temp_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = get_temp_path(path)
    try:
        temp_path.write_text(text, encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def get_shard_indices(num_items: int, num_shards: int, shard_rank: int) -> list[int]:
    return [idx for idx in range(num_items) if idx % num_shards == shard_rank]


def get_generate_batch_size(args: argparse.Namespace, num_items: int) -> int:
    if args.generate_batch_size <= 0:
        return max(num_items, 1)
    return args.generate_batch_size


def iter_batches(items: list[Any], batch_size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def get_summary_group(record: dict[str, Any]) -> str:
    data_source = str(record.get("data_source", "")).strip()
    if data_source:
        return data_source
    return str(record.get("dataset", "")).strip() or "unknown"


def make_dataset_metrics(
    dataset_specs: list[DatasetSpec],
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    del dataset_specs
    metrics: dict[str, dict[str, Any]] = {}
    for record in records:
        dataset_name = get_summary_group(record)
        if dataset_name not in metrics:
            metrics[dataset_name] = {
                "dataset": dataset_name,
                "data_source": dataset_name,
                "path": str(record.get("data_path", "")),
                "num_questions": 0,
                "pass_count": 0,
                "correct_count": 0,
            }
        metrics[dataset_name]["num_questions"] += 1
    return metrics


def to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def jsonable(value: Any) -> Any:
    value = to_python(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def prompt_to_messages(prompt_value: Any) -> list[dict[str, str]]:
    prompt_value = to_python(prompt_value)
    if isinstance(prompt_value, dict):
        prompt_value = [prompt_value]
    if isinstance(prompt_value, list):
        messages = []
        for item in prompt_value:
            item = to_python(item)
            if isinstance(item, dict):
                messages.append(
                    {
                        "role": str(item.get("role", "user")),
                        "content": str(item.get("content", "")),
                    }
                )
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages
    return [{"role": "user", "content": str(prompt_value)}]


def format_prompt(tokenizer: Any, messages: list[dict[str, str]], use_chat_template: bool) -> str:
    if use_chat_template:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            print(f"Falling back to raw prompt because chat template failed: {exc}", flush=True)

    if len(messages) == 1 and messages[0]["role"] == "user":
        return messages[0]["content"]
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def normalize_ground_truth(raw_ground_truth: Any) -> str:
    ground_truth = str(raw_ground_truth).strip()
    boxed = last_boxed_only_string(ground_truth)
    if boxed is not None:
        try:
            return remove_boxed(boxed)
        except Exception:
            pass
    return ground_truth


def get_ground_truth(row: pd.Series) -> str:
    reward_model = to_python(row["reward_model"])
    if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
        raise ValueError(f"Missing reward_model.ground_truth for row {row.name}")
    return normalize_ground_truth(reward_model["ground_truth"])


def score_generation(text: str, ground_truth: str, timeout: float) -> float:
    return float(math_verify.compute_score(text, ground_truth, timeout=timeout))


def load_examples(
    dataset_specs: list[DatasetSpec],
) -> tuple[list[list[dict[str, str]]], list[str], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    prompt_messages: list[list[dict[str, str]]] = []
    ground_truths: list[str] = []
    metadata: list[dict[str, Any]] = []
    dataset_metrics = {
        dataset_spec.name: {
            "dataset": dataset_spec.name,
            "path": dataset_spec.path,
            "num_questions": 0,
            "pass_count": 0,
            "correct_count": 0,
        }
        for dataset_spec in dataset_specs
    }

    for dataset_spec in dataset_specs:
        df = pd.read_parquet(dataset_spec.path)
        if len(df) == 0:
            raise ValueError(f"No examples found in {dataset_spec.path}")

        print(f"Loaded {len(df)} examples from {dataset_spec.name}: {dataset_spec.path}", flush=True)
        dataset_metrics[dataset_spec.name]["num_questions"] = len(df)
        for _, row in df.iterrows():
            example_index = len(metadata)
            messages = prompt_to_messages(row["prompt"])
            prompt_messages.append(messages)
            ground_truths.append(get_ground_truth(row))
            metadata.append(
                {
                    "example_index": example_index,
                    "dataset": dataset_spec.name,
                    "data_path": dataset_spec.path,
                    "row_index": jsonable(row.name),
                    "data_source": str(row.get("data_source", "")),
                    "ability": str(row.get("ability", "")),
                    "prompt": jsonable(messages),
                    "extra_info": jsonable(row.get("extra_info", {})),
                }
            )

    return prompt_messages, ground_truths, metadata, dataset_metrics


def completion_to_raw_record(sample_idx: int, completion: Any) -> dict[str, Any]:
    return {
        "sample_index": sample_idx,
        "text": completion.text,
        "finish_reason": jsonable(getattr(completion, "finish_reason", None)),
        "stop_reason": jsonable(getattr(completion, "stop_reason", None)),
        "token_ids": jsonable(getattr(completion, "token_ids", None)),
        "cumulative_logprob": jsonable(getattr(completion, "cumulative_logprob", None)),
    }


def add_judgment_to_rollout(rollout: dict[str, Any], ground_truth: str, verify_timeout: float) -> dict[str, Any]:
    score = score_generation(str(rollout["text"]), ground_truth, timeout=verify_timeout)
    return {**rollout, "score": score, "correct": score > 0.0}


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "tokenizer": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "seed": args.seed,
    }


def build_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_length,
    )


def generate_single_process(
    args: argparse.Namespace,
    prompt_messages: list[list[dict[str, str]]],
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    raw_outputs: list[list[dict[str, Any]] | None] = [None] * len(prompt_messages)
    formatted_prompts: list[str | None] = [None] * len(prompt_messages)
    for batch_results in generate_single_process_batches(args, prompt_messages):
        for prompt_idx, formatted_prompt, rollouts in batch_results:
            formatted_prompts[prompt_idx] = formatted_prompt
            raw_outputs[prompt_idx] = rollouts

    missing = [idx for idx, output in enumerate(raw_outputs) if output is None]
    if missing:
        raise RuntimeError(f"Missing generation outputs for prompt indices: {missing[:10]}")
    missing_prompts = [idx for idx, prompt in enumerate(formatted_prompts) if prompt is None]
    if missing_prompts:
        raise RuntimeError(f"Missing formatted prompts for prompt indices: {missing_prompts[:10]}")

    return [output for output in raw_outputs if output is not None], [
        prompt for prompt in formatted_prompts if prompt is not None
    ]


def generate_single_process_batches(
    args: argparse.Namespace,
    prompt_messages: list[list[dict[str, str]]],
) -> Iterator[list[tuple[int, str, list[dict[str, Any]]]]]:
    from vllm import LLM

    llm = LLM(**build_llm_kwargs(args))
    tokenizer = llm.get_tokenizer()
    formatted_prompts = [
        (prompt_idx, format_prompt(tokenizer, messages, use_chat_template=not args.no_chat_template))
        for prompt_idx, messages in enumerate(prompt_messages)
    ]
    batch_size = get_generate_batch_size(args, len(formatted_prompts))
    for batch_idx, batch in enumerate(iter_batches(formatted_prompts, batch_size), start=1):
        print(
            f"[single worker] generating batch {batch_idx} "
            f"num_prompts={len(batch)} batch_size={batch_size}",
            flush=True,
        )
        outputs = llm.generate([prompt for _, prompt in batch], build_sampling_params(args))
        yield [
            (
                prompt_idx,
                formatted_prompt,
                [
                    completion_to_raw_record(sample_idx, completion)
                    for sample_idx, completion in enumerate(request_output.outputs)
                ],
            )
            for (prompt_idx, formatted_prompt), request_output in zip(batch, outputs)
        ]


def get_visible_gpu_ids(num_required: int) -> list[str]:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        gpu_ids = [gpu_id.strip() for gpu_id in visible_devices.split(",") if gpu_id.strip()]
    else:
        gpu_ids = [str(gpu_id) for gpu_id in range(num_required)]

    if len(gpu_ids) < num_required:
        raise ValueError(
            f"Need at least {num_required} visible GPUs for tp*dp, but CUDA_VISIBLE_DEVICES={visible_devices!r}"
        )
    return gpu_ids[:num_required]


def split_indexed_prompts(
    prompt_messages: list[list[dict[str, str]]],
    num_shards: int,
) -> list[list[tuple[int, list[dict[str, str]]]]]:
    shards: list[list[tuple[int, list[dict[str, str]]]]] = [[] for _ in range(num_shards)]
    for prompt_idx, messages in enumerate(prompt_messages):
        shards[prompt_idx % num_shards].append((prompt_idx, messages))
    return shards


def dp_worker(
    worker_rank: int,
    args: argparse.Namespace,
    indexed_prompts: list[tuple[int, list[dict[str, str]]]],
    gpu_ids: list[str],
    result_queue: multiprocessing.Queue,
) -> None:
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
        from vllm import LLM

        print(
            f"[dp worker {worker_rank}] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
            f"num_prompts={len(indexed_prompts)}",
            flush=True,
        )
        llm = LLM(**build_llm_kwargs(args))
        tokenizer = llm.get_tokenizer()
        formatted_prompts = [
            (prompt_idx, format_prompt(tokenizer, messages, use_chat_template=not args.no_chat_template))
            for prompt_idx, messages in indexed_prompts
        ]
        batch_size = get_generate_batch_size(args, len(formatted_prompts))
        for batch_idx, batch in enumerate(iter_batches(formatted_prompts, batch_size), start=1):
            print(
                f"[dp worker {worker_rank}] generating batch {batch_idx} "
                f"num_prompts={len(batch)} batch_size={batch_size}",
                flush=True,
            )
            outputs = llm.generate([prompt for _, prompt in batch], build_sampling_params(args))
            worker_results = []
            for (prompt_idx, formatted_prompt), request_output in zip(batch, outputs):
                worker_results.append(
                    (
                        prompt_idx,
                        formatted_prompt,
                        [
                            completion_to_raw_record(sample_idx, completion)
                            for sample_idx, completion in enumerate(request_output.outputs)
                        ],
                    )
                )
            result_queue.put({"rank": worker_rank, "results": worker_results})
        result_queue.put({"rank": worker_rank, "done": True})
    except Exception:
        result_queue.put({"rank": worker_rank, "error": traceback.format_exc()})


def generate_data_parallel(
    args: argparse.Namespace,
    prompt_messages: list[list[dict[str, str]]],
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    raw_outputs: list[list[dict[str, Any]] | None] = [None] * len(prompt_messages)
    formatted_prompts: list[str | None] = [None] * len(prompt_messages)
    for batch_results in generate_data_parallel_batches(args, prompt_messages):
        for prompt_idx, formatted_prompt, rollouts in batch_results:
            formatted_prompts[prompt_idx] = formatted_prompt
            raw_outputs[prompt_idx] = rollouts

    missing = [idx for idx, output in enumerate(raw_outputs) if output is None]
    if missing:
        raise RuntimeError(f"Missing generation outputs for prompt indices: {missing[:10]}")

    missing_prompts = [idx for idx, prompt in enumerate(formatted_prompts) if prompt is None]
    if missing_prompts:
        raise RuntimeError(f"Missing formatted prompts for prompt indices: {missing_prompts[:10]}")

    return [output for output in raw_outputs if output is not None], [
        prompt for prompt in formatted_prompts if prompt is not None
    ]


def generate_data_parallel_batches(
    args: argparse.Namespace,
    prompt_messages: list[list[dict[str, str]]],
) -> Iterator[list[tuple[int, str, list[dict[str, Any]]]]]:
    total_gpus = args.tensor_parallel_size * args.data_parallel_size
    gpu_ids = get_visible_gpu_ids(total_gpus)
    prompt_shards = split_indexed_prompts(prompt_messages, args.data_parallel_size)
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for worker_rank, prompt_shard in enumerate(prompt_shards):
        worker_gpu_ids = gpu_ids[
            worker_rank * args.tensor_parallel_size : (worker_rank + 1) * args.tensor_parallel_size
        ]
        process = ctx.Process(
            target=dp_worker,
            args=(worker_rank, args, prompt_shard, worker_gpu_ids, result_queue),
        )
        process.start()
        processes.append(process)

    worker_errors = []
    completed_workers: set[int] = set()
    while len(completed_workers) < len(processes):
        message = result_queue.get()
        if "error" in message:
            worker_errors.append(f"worker {message['rank']} failed:\n{message['error']}")
            break
        if message.get("done"):
            completed_workers.add(int(message["rank"]))
        else:
            yield message["results"]

    for process in processes:
        process.join()

    failed_exit_codes = [process.exitcode for process in processes if process.exitcode != 0]
    if worker_errors or failed_exit_codes:
        joined_errors = "\n".join(worker_errors)
        raise RuntimeError(f"Data parallel generation failed. exit_codes={failed_exit_codes}\n{joined_errors}")


def generate_rollouts(
    args: argparse.Namespace,
    prompt_messages: list[list[dict[str, str]]],
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    if not prompt_messages:
        return [], []
    if args.data_parallel_size == 1:
        return generate_single_process(args, prompt_messages)
    return generate_data_parallel(args, prompt_messages)


def generate_rollout_batches(
    args: argparse.Namespace,
    prompt_messages: list[list[dict[str, str]]],
) -> Iterator[list[tuple[int, str, list[dict[str, Any]]]]]:
    if not prompt_messages:
        return
    if args.data_parallel_size == 1:
        yield from generate_single_process_batches(args, prompt_messages)
    else:
        yield from generate_data_parallel_batches(args, prompt_messages)


def build_summary(
    args: argparse.Namespace,
    output_path: Path,
    dataset_metrics: dict[str, dict[str, Any]],
    pass_count: int,
    correct_count: int,
) -> dict[str, Any]:
    dataset_summaries = []
    total_questions = 0
    total_samples = 0
    for dataset_name, metric in dataset_metrics.items():
        num_questions = int(metric["num_questions"])
        dataset_total_samples = num_questions * args.n
        total_questions += num_questions
        total_samples += dataset_total_samples
        dataset_summaries.append(
            {
                "dataset": dataset_name,
                "data_source": metric.get("data_source", dataset_name),
                "path": metric["path"],
                "num_questions": num_questions,
                "samples_per_question": args.n,
                f"pass@{args.n}": safe_divide(metric["pass_count"], num_questions),
                f"mean@{args.n}_accuracy": safe_divide(metric["correct_count"], dataset_total_samples),
                "total_correct": metric["correct_count"],
                "total_samples": dataset_total_samples,
            }
        )

    return {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_response_length": args.max_response_length,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "num_shards": args.num_shards,
        "shard_rank": args.shard_rank,
        "enforce_eager": args.enforce_eager,
        "disable_custom_all_reduce": args.disable_custom_all_reduce,
        "n": args.n,
        "output": str(output_path),
        "num_questions": total_questions,
        "samples_per_question": args.n,
        f"pass@{args.n}": safe_divide(pass_count, total_questions),
        f"mean@{args.n}_accuracy": safe_divide(correct_count, total_samples),
        "total_correct": correct_count,
        "total_samples": total_samples,
        "datasets": dataset_summaries,
    }


def print_summary(summary: dict[str, Any], args: argparse.Namespace) -> None:
    print(f"num_questions: {summary['num_questions']}", flush=True)
    print(f"samples_per_question: {args.n}", flush=True)
    print(f"pass@{args.n}: {summary[f'pass@{args.n}']:.6f}", flush=True)
    print(f"mean@{args.n} accuracy: {summary[f'mean@{args.n}_accuracy']:.6f}", flush=True)
    print(f"total_correct: {summary['total_correct']}/{summary['total_samples']}", flush=True)
    for dataset_summary in summary["datasets"]:
        print(
            f"{dataset_summary['dataset']} pass@{args.n}: {dataset_summary[f'pass@{args.n}']:.6f}, "
            f"mean@{args.n} accuracy: {dataset_summary[f'mean@{args.n}_accuracy']:.6f}, "
            f"total_correct: {dataset_summary['total_correct']}/{dataset_summary['total_samples']}",
            flush=True,
        )


def write_records_and_summary(
    args: argparse.Namespace,
    output_path: Path,
    summary_output_path: Path,
    dataset_specs: list[DatasetSpec],
    raw_outputs: list[list[dict[str, Any]]],
    formatted_prompts: list[str],
    ground_truths: list[str],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_metrics = make_dataset_metrics(dataset_specs, metadata)

    pass_count = 0
    correct_count = 0
    temp_output_path = get_temp_path(output_path)
    try:
        with temp_output_path.open("w", encoding="utf-8") as f:
            for local_idx, (raw_rollouts, ground_truth) in enumerate(zip(raw_outputs, ground_truths)):
                metadata[local_idx]["formatted_prompt"] = formatted_prompts[local_idx]
                rollouts = [
                    add_judgment_to_rollout(rollout, ground_truth=ground_truth, verify_timeout=args.verify_timeout)
                    for rollout in raw_rollouts
                ]
                example_correct_count = sum(int(rollout["correct"]) for rollout in rollouts)
                dataset_name = get_summary_group(metadata[local_idx])
                dataset_metrics[dataset_name]["pass_count"] += int(example_correct_count > 0)
                dataset_metrics[dataset_name]["correct_count"] += example_correct_count

                correct_count += example_correct_count
                pass_count += int(example_correct_count > 0)
                record = {
                    **metadata[local_idx],
                    "ground_truth": ground_truth,
                    "num_rollouts": len(raw_rollouts),
                    "correct_count": example_correct_count,
                    f"pass@{args.n}": example_correct_count > 0,
                    "rollouts": rollouts,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[{local_idx + 1}/{len(metadata)}] {dataset_name} "
                    f"example_index={metadata[local_idx]['example_index']} "
                    f"correct={example_correct_count}/{len(raw_rollouts)}",
                    flush=True,
                )
        os.replace(temp_output_path, output_path)
    finally:
        if temp_output_path.exists():
            temp_output_path.unlink()

    summary = build_summary(args, output_path, dataset_metrics, pass_count, correct_count)
    atomic_write_text(summary_output_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def write_streamed_records_and_summary(
    args: argparse.Namespace,
    output_path: Path,
    summary_output_path: Path,
    dataset_specs: list[DatasetSpec],
    prompt_messages: list[list[dict[str, str]]],
    ground_truths: list[str],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_metrics = make_dataset_metrics(dataset_specs, metadata)

    pass_count = 0
    correct_count = 0
    seen_indices: set[int] = set()
    temp_output_path = get_temp_path(output_path)
    try:
        with temp_output_path.open("w", encoding="utf-8") as f:
            for batch_idx, batch_results in enumerate(generate_rollout_batches(args, prompt_messages), start=1):
                print(
                    f"[verify] batch {batch_idx} num_prompts={len(batch_results)} "
                    f"processed={len(seen_indices)}/{len(metadata)}",
                    flush=True,
                )
                for local_idx, formatted_prompt, raw_rollouts in batch_results:
                    if local_idx in seen_indices:
                        raise RuntimeError(f"Duplicate generation output for local prompt index {local_idx}")
                    seen_indices.add(local_idx)

                    metadata[local_idx]["formatted_prompt"] = formatted_prompt
                    ground_truth = ground_truths[local_idx]
                    rollouts = [
                        add_judgment_to_rollout(rollout, ground_truth=ground_truth, verify_timeout=args.verify_timeout)
                        for rollout in raw_rollouts
                    ]
                    example_correct_count = sum(int(rollout["correct"]) for rollout in rollouts)
                    dataset_name = get_summary_group(metadata[local_idx])
                    dataset_metrics[dataset_name]["pass_count"] += int(example_correct_count > 0)
                    dataset_metrics[dataset_name]["correct_count"] += example_correct_count

                    correct_count += example_correct_count
                    pass_count += int(example_correct_count > 0)
                    record = {
                        **metadata[local_idx],
                        "ground_truth": ground_truth,
                        "num_rollouts": len(raw_rollouts),
                        "correct_count": example_correct_count,
                        f"pass@{args.n}": example_correct_count > 0,
                        "rollouts": rollouts,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    print(
                        f"[{len(seen_indices)}/{len(metadata)}] {dataset_name} "
                        f"example_index={metadata[local_idx]['example_index']} "
                        f"correct={example_correct_count}/{len(raw_rollouts)}",
                        flush=True,
                    )

        missing = sorted(set(range(len(metadata))) - seen_indices)
        if missing:
            raise RuntimeError(f"Missing generation outputs for local prompt indices: {missing[:10]}")
        os.replace(temp_output_path, output_path)
    finally:
        if temp_output_path.exists():
            temp_output_path.unlink()

    summary = build_summary(args, output_path, dataset_metrics, pass_count, correct_count)
    atomic_write_text(summary_output_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def wait_for_shard_outputs(shard_paths: list[Path], min_mtime: float, timeout: float) -> None:
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    last_report = 0.0
    while True:
        missing = []
        for shard_path in shard_paths:
            if not shard_path.exists() or shard_path.stat().st_mtime < min_mtime:
                missing.append(shard_path)
        if not missing:
            return

        now = time.monotonic()
        if deadline is not None and now >= deadline:
            missing_names = ", ".join(str(path) for path in missing[:10])
            raise TimeoutError(f"Timed out waiting for shard outputs: {missing_names}")
        if now - last_report >= 30:
            missing_names = ", ".join(str(path) for path in missing[:3])
            print(
                f"Waiting for {len(missing)}/{len(shard_paths)} shard outputs. "
                f"Missing: {missing_names}",
                flush=True,
            )
            last_report = now
        time.sleep(5)


def merge_shard_outputs(
    args: argparse.Namespace,
    output_path: Path,
    summary_output_path: Path,
    dataset_specs: list[DatasetSpec],
    expected_total: int,
    min_mtime: float,
) -> dict[str, Any]:
    shard_paths = [get_shard_path(output_path, shard_rank, args.num_shards) for shard_rank in range(args.num_shards)]
    wait_for_shard_outputs(shard_paths, min_mtime=min_mtime, timeout=args.merge_timeout)

    records: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        records.extend(read_jsonl_records(shard_path))
    records.sort(key=lambda record: int(record["example_index"]))

    seen_indices: set[int] = set()
    duplicate_indices: set[int] = set()
    for record in records:
        example_index = int(record["example_index"])
        if example_index in seen_indices:
            duplicate_indices.add(example_index)
        seen_indices.add(example_index)
    if duplicate_indices:
        raise RuntimeError(f"Duplicate example_index values in shard outputs: {sorted(duplicate_indices)[:10]}")
    if len(records) != expected_total:
        raise RuntimeError(f"Merged {len(records)} records, expected {expected_total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = get_temp_path(output_path)
    try:
        with temp_output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temp_output_path, output_path)
    finally:
        if temp_output_path.exists():
            temp_output_path.unlink()

    dataset_metrics = make_dataset_metrics(dataset_specs, records)
    pass_count = 0
    correct_count = 0
    for record in records:
        dataset_name = get_summary_group(record)
        example_correct_count = int(record["correct_count"])
        pass_count += int(example_correct_count > 0)
        correct_count += example_correct_count
        dataset_metrics[dataset_name]["pass_count"] += int(example_correct_count > 0)
        dataset_metrics[dataset_name]["correct_count"] += example_correct_count

    summary = build_summary(args, output_path, dataset_metrics, pass_count, correct_count)
    summary["shard_outputs"] = [str(path) for path in shard_paths]
    atomic_write_text(summary_output_path, json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    run_started_at = time.time()
    args = parse_args()
    dataset_specs = get_dataset_specs(args.data)

    if args.n <= 0:
        raise ValueError(f"--n must be positive, got {args.n}")
    if args.tensor_parallel_size <= 0:
        raise ValueError(f"--tensor-parallel-size/--tp must be positive, got {args.tensor_parallel_size}")
    if args.data_parallel_size <= 0:
        raise ValueError(f"--data-parallel-size/--dp must be positive, got {args.data_parallel_size}")
    if args.num_shards <= 0:
        raise ValueError(f"--num-shards must be positive, got {args.num_shards}")
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        raise ValueError(f"--shard-rank must be in [0, {args.num_shards}), got {args.shard_rank}")

    prompt_messages, ground_truths, metadata, _ = load_examples(dataset_specs)

    output_path = get_output_path(args)
    summary_output_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(
        output_path.suffix + ".summary.json"
    )

    shard_indices = get_shard_indices(len(prompt_messages), args.num_shards, args.shard_rank)
    if args.num_shards > 1:
        print(
            f"Shard {args.shard_rank}/{args.num_shards}: "
            f"{len(shard_indices)}/{len(prompt_messages)} examples",
            flush=True,
        )

    shard_prompt_messages = [prompt_messages[idx] for idx in shard_indices]
    shard_ground_truths = [ground_truths[idx] for idx in shard_indices]
    shard_metadata = [dict(metadata[idx]) for idx in shard_indices]

    write_output_path = output_path
    write_summary_output_path = summary_output_path
    if args.num_shards > 1:
        write_output_path = get_shard_path(output_path, args.shard_rank, args.num_shards)
        write_summary_output_path = get_shard_path(summary_output_path, args.shard_rank, args.num_shards)

    summary = write_streamed_records_and_summary(
        args=args,
        output_path=write_output_path,
        summary_output_path=write_summary_output_path,
        dataset_specs=dataset_specs,
        prompt_messages=shard_prompt_messages,
        ground_truths=shard_ground_truths,
        metadata=shard_metadata,
    )

    print(f"Rollouts and judgments saved to {write_output_path}", flush=True)
    print(f"Summary saved to {write_summary_output_path}", flush=True)
    print_summary(summary, args)

    if args.num_shards > 1 and args.merge_shards and args.shard_rank == 0:
        print("Rank 0 waiting for shard outputs before final merge...", flush=True)
        merged_summary = merge_shard_outputs(
            args=args,
            output_path=output_path,
            summary_output_path=summary_output_path,
            dataset_specs=dataset_specs,
            expected_total=len(prompt_messages),
            min_mtime=run_started_at,
        )
        print(f"Merged rollouts and judgments saved to {output_path}", flush=True)
        print(f"Merged summary saved to {summary_output_path}", flush=True)
        print_summary(merged_summary, args)


if __name__ == "__main__":
    main()
