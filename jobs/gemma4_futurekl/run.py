#!/usr/bin/env python3
"""Launch one Gemma 4 E2B text-only RL run on a local H200 node."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERL = REPO / "verl"
REWARD = REPO / "jobs/gemma4_data/reward.py"
PROFILES = {
    # steps, prompt limit, response limit, prompts per step, prompts per minibatch, group size, eval samples
    "smoke-short": (2, 1024, 512, 16, 8, 2, 2),
    "smoke-long": (2, 2048, 8192, 16, 8, 2, 1),
    "pilot": (50, 2048, 8192, 256, 64, 5, 8),
}
DEFAULT_BETA = {"grpo": 0, "loss-kl": 0.001, "frpo": 0.01, "dapo": 0, "dapo-frpo": 0.01}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("arm", choices=DEFAULT_BETA)
    parser.add_argument("--model", type=Path, required=True, help="Staged Gemma 4 E2B-it model directory")
    parser.add_argument("--train", type=Path, required=True, help="Decontaminated training parquet")
    parser.add_argument("--validation", type=Path, action="append", required=True, help="Validation parquet, repeatable")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, help="KL coefficient for the chosen arm")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--response-limit", type=int)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--val-n", type=int)
    parser.add_argument("--wandb", action="store_true", help="Also log to Weights & Biases")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-command", action="store_true", help="Print the command without importing the training stack")
    mode.add_argument("--resolve", action="store_true", help="Print the resolved Hydra config without starting Ray")
    parser.add_argument("--override", action="append", default=[], help="Additional Hydra override, repeatable")
    args = parser.parse_args()
    profile_keys = ("steps", "prompt_limit", "response_limit", "train_batch_size", "minibatch_size", "rollouts", "val_n")
    for name, default in zip(profile_keys, PROFILES[args.profile], strict=True):
        if getattr(args, name) is None:
            setattr(args, name, default)
    if args.beta is None:
        args.beta = DEFAULT_BETA[args.arm]
    if any(getattr(args, name) < 1 for name in (*profile_keys, "gpus")):
        parser.error("Steps, lengths, batch sizes, rollout counts, and GPU counts must be positive.")
    if args.rollouts < 2:
        parser.error("GRPO needs at least two rollouts per prompt.")
    if args.train_batch_size % args.minibatch_size or args.minibatch_size * args.rollouts % args.gpus:
        parser.error("Use whole minibatches and split their trajectories evenly over GPUs.")
    if not math.isfinite(args.beta) or (args.beta != 0 if not DEFAULT_BETA[args.arm] else args.beta <= 0):
        parser.error("Use a finite positive beta for a KL arm or zero beta for an unregularized arm.")
    if args.profile == "smoke-long" and args.arm.startswith("dapo"):
        parser.error("Use a GRPO arm for the forced-length smoke. DAPO requires mixed verifier outcomes.")
    args.model = args.model.resolve()
    args.train = args.train.resolve()
    args.validation = [path.resolve() for path in args.validation]
    args.output = args.output.resolve()
    return args


def hydra_value(value):
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, list):
        return "[" + ",".join(hydra_value(item) for item in value) + "]"
    return json.dumps(value)


def command(args):
    smoke = args.profile != "pilot"
    forced_length = args.profile == "smoke-long"
    dapo = args.arm.startswith("dapo")
    frpo = args.arm in {"frpo", "dapo-frpo"}
    max_tokens = args.prompt_limit + args.response_limit
    overrides = {
        "data.train_files": args.train,
        "data.val_files": args.validation,
        "data.train_batch_size": args.train_batch_size,
        "data.max_prompt_length": args.prompt_limit,
        "data.max_response_length": args.response_limit,
        "data.truncation": "error",
        "data.filter_overlong_prompts": True,
        "data.filter_overlong_prompts_workers": 4,
        # Python 3.14 uses forkserver for DataLoader workers. RLHFDataset omits
        # its Arrow table when pickled, so load batches in the trainer process.
        "data.dataloader_num_workers": 0,
        "data.train_max_samples": args.train_batch_size * 4 if smoke and not dapo else -1,
        "data.val_max_samples": 4 if smoke else -1,
        "data.seed": args.seed,
        "data.return_raw_chat": True,
        "data.return_multi_modal_inputs": False,
        "+data.apply_chat_template_kwargs.enable_thinking": True,
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.use_kl_in_reward": False,
        "prm.eta": 0.0,
        "prm.force_ref_policy": True,
        "prm.log_stats": True,
        "prm.stats_windows": [1, 64, 512],
        "actor_rollout_ref.model.path": args.model,
        "+actor_rollout_ref.model.override_config.attn_implementation": "sdpa",
        "actor_rollout_ref.model.use_remove_padding": False,
        "actor_rollout_ref.model.use_fused_kernels": False,
        "actor_rollout_ref.model.use_liger": False,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.actor.fsdp_config.seed": args.seed,
        "actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking": True,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_mini_batch_size": args.minibatch_size,
        "actor_rollout_ref.actor.data_loader_seed": args.seed,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.clip_ratio_low": 0.2,
        "actor_rollout_ref.actor.clip_ratio_high": 0.28 if dapo else 0.2,
        "actor_rollout_ref.actor.clip_ratio_c": 10.0 if dapo else 3.0,
        "actor_rollout_ref.actor.use_kl_loss": args.arm == "loss-kl",
        "actor_rollout_ref.actor.kl_loss_coef": args.beta if args.arm == "loss-kl" else 0.0,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "prl_current_kl" if frpo else "vanilla",
        "actor_rollout_ref.actor.policy_loss.prl_eta": 1.0 / args.beta if frpo else 0.0,
        "actor_rollout_ref.actor.policy_loss.prl_kl_estimator": "reverse_k1",
        "actor_rollout_ref.actor.policy_loss.prl_future_window": -1,
        "actor_rollout_ref.actor.policy_loss.prl_kl_scope": "future",
        "actor_rollout_ref.ref.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.ref.fsdp_config.seed": args.seed,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.dtype": "bfloat16",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.55,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.max_model_len": max_tokens,
        "actor_rollout_ref.rollout.max_num_batched_tokens": max_tokens,
        "actor_rollout_ref.rollout.max_num_seqs": 64,
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes": 9216,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.logprobs_mode": "raw_logprobs",
        "++actor_rollout_ref.rollout.seed": args.seed,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.n": args.rollouts,
        "actor_rollout_ref.rollout.val_kwargs.n": args.val_n,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        "reward.reward_manager.source": "importlib",
        "reward.reward_manager.name": "GemmaMathRewardManager",
        "reward.reward_manager.module.path": REWARD,
        "critic.enable": False,
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": args.gpus,
        "trainer.use_legacy_worker_impl": "disable",
        "trainer.total_training_steps": args.steps,
        "trainer.total_epochs": max(30, args.steps),
        "trainer.resume_mode": "disable",
        "trainer.val_before_train": not forced_length,
        "trainer.test_freq": -1 if smoke else 10,
        "trainer.save_freq": args.steps if smoke else 10,
        "trainer.default_local_dir": args.output / "checkpoints",
        "trainer.validation_data_dir": args.output / "validation",
        "trainer.logger": ["console", "wandb"] if args.wandb else ["console"],
        "trainer.project_name": "gemma4-futurekl",
        "trainer.experiment_name": args.output.name,
        "+ray_kwargs.ray_init.address": "local",
        "+ray_kwargs.ray_init.include_dashboard": False,
        "+ray_kwargs.ray_init._temp_dir": "/tmp/gemma4_" + hashlib.sha256(os.fsencode(args.output)).hexdigest()[:10],
        "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONHASHSEED": "0",
    }
    if forced_length:
        overrides.update({
            "actor_rollout_ref.rollout.agent.default_agent_loop": "gemma_capacity_probe",
            "actor_rollout_ref.rollout.agent.agent_loop_config_path": HERE / "capacity.yaml",
        })
    if dapo:
        overrides.update({
            "data.gen_batch_size": 2 * args.train_batch_size,
            "algorithm.filter_groups.enable": True,
            "algorithm.filter_groups.metric": "acc",
            "algorithm.filter_groups.max_num_gen_batches": 10,
            "reward.reward_kwargs.overlong_buffer_cfg.enable": True,
            "reward.reward_kwargs.overlong_buffer_cfg.len": min(512, args.response_limit // 4),
            "reward.reward_kwargs.overlong_buffer_cfg.penalty_factor": 1.0,
            "reward.reward_kwargs.overlong_buffer_cfg.log": True,
            "reward.reward_kwargs.max_resp_len": args.response_limit,
        })
    entrypoint = "recipe.dapo.main_dapo" if dapo else "verl.trainer.main_ppo"
    return [sys.executable, "-m", entrypoint, *(f"{key}={hydra_value(value)}" for key, value in overrides.items()), *args.override]


def environment(args):
    env = dict(os.environ)
    for key in ("RAY_ADDRESS", "RAY_NAMESPACE", "RAY_JOB_ID", "WANDB_RUN_ID", "WANDB_RESUME"):
        env.pop(key, None)
    python_paths = [str(VERL), str(HERE)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env.update({
        "PYTHONPATH": os.pathsep.join(python_paths),
        # Gemma registers RoPE buffers through a set. FSDP1 broadcasts them in registration order.
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(args.gpus)),
        "WANDB_DIR": str(args.output),
    })
    return env


def check_inputs(args):
    for path in (args.model / "config.json", args.train, *args.validation, REWARD):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads((args.model / "config.json").read_text())
    if config.get("model_type") != "gemma4" or config.get("text_config", {}).get("num_hidden_layers") != 35:
        raise ValueError(f"Expected a staged Gemma 4 E2B model at {args.model}")


def check_idle(gpus):
    query = ["nvidia-smi", "--format=csv,noheader,nounits"]
    names = subprocess.check_output([*query, "--query-gpu=name"], text=True).splitlines()
    if len(names) < gpus or any("H200" not in name for name in names[:gpus]):
        raise RuntimeError(f"Expected {gpus} NVIDIA H200 GPUs, found {names}.")
    processes = subprocess.check_output([*query, "--query-compute-apps=pid"], text=True).strip()
    if processes:
        raise RuntimeError(f"The node already has GPU compute processes: {processes}")


def write_status(path, status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def launch(args, cmd, env):
    check_inputs(args)
    with Path("/tmp/gemma4-futurekl.node.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another Gemma 4 run holds the node lock.") from error
        check_idle(args.gpus)
        resolved = subprocess.check_output([*cmd, "--cfg", "job", "--resolve"], cwd=VERL, env=env, text=True)
        # FlashAttention 4 prints startup diagnostics to stdout before Hydra's YAML.
        resolved = "\n".join(line for line in resolved.splitlines() if not line.startswith("[FA4]")) + "\n"
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.yaml").write_text(resolved)
        (args.output / "command.txt").write_text(shlex.join(cmd) + "\n")
        versions = {package: importlib.metadata.version(package) for package in ("torch", "transformers", "vllm", "ray")}
        status = {"host": socket.gethostname(), "start": timestamp(), "profile": args.profile, "arm": args.arm,
                  "seed": args.seed, "beta": args.beta, "versions": versions, "state": "starting"}
        status_path = args.output / "status.json"
        write_status(status_path, status)
        print(f"Logging to {args.output / 'train.log'}", flush=True)
        with (args.output / "train.log").open("w") as log:
            process = subprocess.Popen(cmd, cwd=VERL, env=env, stdout=log, stderr=subprocess.STDOUT)
            previous = {sig: signal.signal(sig, lambda signum, _: process.send_signal(signum))
                        for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                status.update(state="running", pid=process.pid)
                write_status(status_path, status)
                exit_code = process.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        status.update(state="finished" if exit_code == 0 else "failed", end=timestamp(), exit_code=exit_code)
        write_status(status_path, status)
        return exit_code if exit_code >= 0 else 128 - exit_code


def main():
    args = parse_args()
    cmd = command(args)
    env = environment(args)
    if args.print_command:
        print(shlex.join(cmd))
        return 0
    if args.resolve:
        return subprocess.call([*cmd, "--cfg", "job", "--resolve"], cwd=VERL, env=env)
    return launch(args, cmd, env)


if __name__ == "__main__":
    sys.exit(main())
