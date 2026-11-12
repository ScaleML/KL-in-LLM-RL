# Gemma 4 E2B-it Future-KL pilots

Each pilot uses one node with eight H200 GPUs. All three arms initialize from `google/gemma-4-E2B-it` at revision `4adf15748de9e1a0097b6274cee9c15836c41c37` and use the same frozen reference. The original `model.safetensors` has SHA-256 `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550`.

## Data

Place the three input Parquets at the paths relative to `--data-root` in the [data manifest](../gemma4_data/manifest.json). The training input is the pre-filtered `Jiawei415/DPAO_filter` split from revision `d4bb0f1f83c32cfa06424eee48bf0e4b90990ebd` with Math500 overlaps removed. The preparation script verifies the input hashes, excludes normalized prompt overlaps with the two test sets, and holds out 512 dev prompts with seed 42.
The recorded Parquet checksums reproduce with PyArrow `21.0.0`.

```bash
python jobs/gemma4_data/prepare_data.py --data-root /path/to/pinned-inputs
```

## Recorded pilot settings

| Run | Arm | β | Seed |
| --- | --- | ---: | ---: |
| `gemma4-pilot-grpo-s42-v3` | `grpo` | 0 | 42 |
| `gemma4-pilot-loss-kl-b1e-3-s42` | `loss-kl` | 0.001 | 42 |
| `gemma4-pilot-frpo-b1e-2-s42` | `frpo` | 0.01 | 42 |

Use the matching arm, coefficient, and run name for each row. The recorded pilots used the `pilot` profile with these two overrides and console logging.

```bash
python jobs/gemma4_futurekl/run.py pilot frpo \
  --beta 0.01 --seed 42 \
  --model /path/to/staged/gemma-4-E2B-it \
  --train jobs/gemma4_data/prepared/train.parquet \
  --validation jobs/gemma4_data/prepared/dev.parquet \
  --output /path/to/runs/gemma4-pilot-frpo-b1e-2-s42 \
  --override +actor_rollout_ref.rollout.engine_kwargs.vllm.skip_mm_profiling=true \
  --override trainer.val_before_train=false
```

The pilots were configured for 50 steps, 256 prompts per step, five rollouts per prompt, a minibatch of 64 prompts, and 2,048 prompt / 8,192 response tokens. Dev evaluation uses eight samples per prompt. Dev evaluation and checkpointing run every ten steps. The launcher saves the resolved configuration, command, training log, and status in the output directory. `--print-command` prints the command, and `--resolve` resolves its Hydra configuration without starting Ray.

`matrix.tsv` includes the coefficient sweep and smoke profiles. `smoke-short` checks the training path. `smoke-long` forces generations to the configured response limit for capacity testing. `jobs/gemma4_data/quick_eval.py` runs sampled and greedy evaluation on one GPU with the checkout's `verl` directory on `PYTHONPATH`.

## Recorded training environment

The H200 pilots used Python `3.14.6`, PyTorch `2.13.0`, Transformers `5.6.0.dev0`, vLLM `0.26.1.dev848+g29657ea37.cu133`, Ray `2.51.1+98ef3ef1`, TensorDict `0.12.0`, PyArrow `22.0.0`, and Math-Verify `0.8.0`. Hydra `1.3.2` and OmegaConf `2.3.0` were supplied through an isolated dependency directory on `PYTHONPATH`. The launcher preserves the caller's `PYTHONPATH` and fixes `PYTHONHASHSEED` for consistent Gemma RoPE buffer registration across FSDP workers.
