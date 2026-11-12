#!/usr/bin/env python3
"""Evaluate one Gemma checkpoint with sampled and greedy decoding on one GPU."""

import argparse
import json
import math
from pathlib import Path


def pass_at_k(correct, total, k):
    return 1.0 - math.comb(total - correct, k) / math.comb(total, k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-response-tokens", type=int, default=8192)
    args = parser.parse_args()
    if args.samples < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Sample and prompt counts must be positive")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")

    import pyarrow.parquet as pq
    from transformers import AutoTokenizer, GenerationConfig
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from reward import score_response, warm_verifier

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = GenerationConfig.from_pretrained(args.model)
    eos_ids = config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    rows = pq.read_table(args.data).slice(0, args.limit).to_pylist()
    if not rows:
        parser.error("No prompts in evaluation data")
    prompts = []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True
        )
        if len(prompt_ids) > args.max_prompt_tokens:
            raise ValueError(f"Prompt exceeds the {args.max_prompt_tokens}-token evaluation budget")
        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))

    warm_verifier()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=args.max_prompt_tokens + args.max_response_tokens,
        seed=args.seed,
    )
    stats = {"prompts": len(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output_file:
        for mode, n, temperature in (("sampled", args.samples, 1.0), ("greedy", 1, 0.0)):
            sampling = SamplingParams(
                n=n,
                temperature=temperature,
                top_p=1.0,
                top_k=-1,
                stop_token_ids=eos_ids,
                max_tokens=args.max_response_tokens,
                seed=args.seed,
            )
            generations = llm.generate(prompts, sampling)
            correct = []
            with_final = 0
            truncated = 0
            for index, (row, generation) in enumerate(zip(rows, generations, strict=True)):
                if len(generation.outputs) != n:
                    raise RuntimeError(f"Expected {n} samples for problem {index}")
                solved = 0
                for item in generation.outputs:
                    response_ids = list(item.token_ids)
                    raw_text = tokenizer.decode(response_ids, skip_special_tokens=False)
                    info = score_response(raw_text, row["reward_model"]["ground_truth"])
                    solved += int(info["acc"] > 0)
                    with_final += int(info["has_final"])
                    truncated += int(item.finish_reason == "length")
                    record = {
                        "uid": str(row["extra_info"]["index"]),
                        "data_source": row["data_source"],
                        "mode": mode,
                        "sample": item.index,
                        "acc": info["acc"],
                        "has_final": info["has_final"],
                        "finish_reason": item.finish_reason,
                        "response_token_ids": response_ids,
                        "response": raw_text,
                    }
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                correct.append(solved)
            output_file.flush()
            stats[f"{mode}/has_final"] = with_final / (len(rows) * n)
            stats[f"{mode}/truncated"] = truncated / (len(rows) * n)
            if mode == "greedy":
                stats["greedy/accuracy"] = sum(correct) / len(rows)
            else:
                stats["sampled/mixed_group_fraction"] = sum(0 < c < n for c in correct) / len(rows)
                for k in sorted({1, n, *[k for k in (8, 16) if k <= n]}):
                    stats[f"sampled/pass@{k}"] = sum(pass_at_k(c, n, k) for c in correct) / len(rows)
            print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
