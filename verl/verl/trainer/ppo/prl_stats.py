# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Batch-level bias / variance / correlation statistics of PRL (FRPO) future-KL estimators.

For every valid response token t we form windowed future-KL estimates

    x_W(t) = sum_{j=t}^{min(t+W-1, T)} kl_j        (W tokens, current token included)

and the full-suffix estimate x_inf(t) = sum_{j>=t} kl_j that FRPO uses as the per-token
credit-assignment term, plus x_seq(t) = sum_j kl_j, the KL of the whole response (past + current +
future; identical for every token of a response), which the "full KL" variant uses.  Treating x_inf as the reference, the error of the windowed estimator
decomposes as

    E[(x_W - x_inf)^2] = Var(x_W - x_inf) + (E[x_W - x_inf])^2   (mse = variance + bias^2)

which is what ``bias2_vs_full`` reports (computed as ``mse - var`` and identical to ``mean(d)^2``).
We also log Var(x_W), the Pearson correlation of every window with the full future-KL and with
the current-token KL (x_1), and the full window-by-window correlation matrix.

All quantities are computed on the driver from the batch tensors ``old_log_probs`` (the sampling
policy pi_old) and ``ref_log_prob`` (pi_ref), over all valid response tokens in the batch.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

DEFAULT_STATS_WINDOWS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_STATS_ESTIMATORS: tuple[str, ...] = ("k1",)


def token_kl_estimate(logprob: torch.Tensor, ref_logprob: torch.Tensor, estimator: str) -> torch.Tensor:
    """Per-token single-sample estimates of KL(pi || pi_ref) for a token sampled from pi.

    With r = log pi_ref - log pi:  k1 = -r,  k2 = r^2 / 2,  k3 = exp(r) - r - 1.
    """
    r = ref_logprob - logprob
    if estimator == "k1":
        return -r
    if estimator == "k2":
        return 0.5 * r.square()
    if estimator == "k3":
        r = torch.clamp(r, min=-20.0, max=20.0)
        return torch.exp(r) - r - 1.0
    raise ValueError(f"Unsupported token KL estimator for prl stats: {estimator}")


def windowed_suffix_sums(
    x: torch.Tensor, mask: torch.Tensor, windows: Iterable[int]
) -> dict[float | int, torch.Tensor]:
    """Inclusive windowed suffix sums of ``x`` along the last dim.

    Returns ``{W: sum_{j=t}^{min(t+W-1, T)} x_j}`` for every window ``W`` (the window counts the
    current token) plus ``{math.inf: sum_{j>=t} x_j}``.  Positions outside ``mask`` contribute 0, so
    windows are automatically truncated at the end of each response.
    """
    xm = x * mask
    csum = xm.cumsum(dim=-1)  # C_t = sum_{j<=t}
    total = csum[..., -1:]
    excl = csum - xm  # sum_{j<t}
    out: dict[float | int, torch.Tensor] = {math.inf: total - excl}
    seq_len = x.shape[-1]
    for w in windows:
        w = int(w)
        if w <= 0:
            raise ValueError(f"stats window must be positive, got {w}")
        if w >= seq_len:
            out[w] = out[math.inf]
            continue
        # C_{min(t+w-1, T-1)} - C_{t-1}
        shifted = torch.cat([csum[..., w - 1 :], total.expand(*total.shape[:-1], w - 1)], dim=-1)
        out[w] = shifted - excl
    return out


SEQ_KEY = "seq"  # whole-sequence KL (past + current + future)


def _window_name(w) -> str:
    if w == SEQ_KEY:
        return "seq"
    return "inf" if w == math.inf else f"w{int(w)}"


@torch.no_grad()
def compute_future_kl_stats_from_tensors(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    response_mask: torch.Tensor,
    windows: Iterable[int] = DEFAULT_STATS_WINDOWS,
    estimators: Iterable[str] = DEFAULT_STATS_ESTIMATORS,
    prefix: str = "prl_stats",
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute the future-KL bias/variance/correlation statistics; see the module docstring."""
    logprob = logprob.detach().float()
    ref_logprob = ref_logprob.detach().float()
    mask = response_mask.detach().to(logprob.dtype)
    valid = mask.bool()
    n_valid = int(valid.sum().item())
    metrics: dict[str, float] = {f"{prefix}/num_tokens": float(n_valid)}
    if n_valid < 2:
        return metrics

    windows = sorted({int(w) for w in windows})
    keys: list = [*windows, math.inf, SEQ_KEY]
    names = [_window_name(k) for k in keys]
    full_idx = len(windows)  # index of the full future-KL (x_inf) column

    for est in estimators:
        kl = token_kl_estimate(logprob, ref_logprob, est) * mask
        sums = windowed_suffix_sums(kl, mask, windows)
        sums[SEQ_KEY] = kl.sum(dim=-1, keepdim=True).expand_as(kl) * mask  # whole-sequence KL at every token
        x = torch.stack([sums[k][valid] for k in keys]).double()  # (K, N)
        cur = kl[valid].double()  # current-token KL, x_1
        full = x[full_idx]

        mean = x.mean(dim=1)
        var = x.var(dim=1, unbiased=False)
        diff = x - full  # x_W - x_inf
        mse = diff.square().mean(dim=1)
        var_diff = diff.var(dim=1, unbiased=False)
        bias = diff.mean(dim=1)
        bias2 = mse - var_diff  # == bias^2 ; the user's identity (E[d])^2 = E[d^2] - Var(d)

        xc = x - mean[:, None]
        std = xc.square().mean(dim=1).sqrt()
        corr = (xc @ xc.T) / x.shape[1] / (std[:, None] * std[None, :] + eps)
        cur_c = cur - cur.mean()
        cur_std = cur_c.square().mean().sqrt()
        corr_cur = (cur_c[None, :] * xc).mean(dim=1) / (cur_std * std + eps)

        p = f"{prefix}/{est}"
        metrics[f"{p}/token_kl/mean"] = cur.mean().item()
        metrics[f"{p}/token_kl/var"] = cur.var(unbiased=False).item()
        full_var = var[full_idx].item()
        for i, (k, name) in enumerate(zip(keys, names, strict=True)):
            metrics[f"{p}/{name}/mean"] = mean[i].item()
            metrics[f"{p}/{name}/var"] = var[i].item()
            metrics[f"{p}/{name}/std"] = math.sqrt(max(var[i].item(), 0.0))
            metrics[f"{p}/{name}/var_ratio_to_full"] = var[i].item() / (full_var + eps)
            metrics[f"{p}/{name}/corr_with_token_kl"] = corr_cur[i].item()
            if k != math.inf:
                metrics[f"{p}/{name}/mse_vs_full"] = mse[i].item()
                metrics[f"{p}/{name}/var_diff_vs_full"] = var_diff[i].item()
                metrics[f"{p}/{name}/bias_vs_full"] = bias[i].item()
                metrics[f"{p}/{name}/bias2_vs_full"] = bias2[i].item()
                metrics[f"{p}/{name}/bias2_over_mse"] = bias2[i].item() / (mse[i].item() + eps)
                metrics[f"{p}/{name}/corr_with_full"] = corr[i, full_idx].item()
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                metrics[f"{p}/corr_matrix/{names[i]}__{names[j]}"] = corr[i, j].item()
    return metrics


def compute_future_kl_stats(batch, prm_config: Any = None, prefix: str = "prl_stats") -> dict[str, float]:
    """Entry point used by the trainer: reads tensors and knobs (``prm.stats_windows`` /
    ``prm.stats_estimators``) and returns a flat metrics dict."""
    if "old_log_probs" not in batch.batch or "ref_log_prob" not in batch.batch:
        return {}
    windows = DEFAULT_STATS_WINDOWS
    estimators = DEFAULT_STATS_ESTIMATORS
    if prm_config is not None:
        cfg_windows = prm_config.get("stats_windows", None)
        if cfg_windows:
            windows = [int(w) for w in cfg_windows]
        cfg_est = prm_config.get("stats_estimators", None)
        if cfg_est:
            estimators = [str(e) for e in cfg_est]
    return compute_future_kl_stats_from_tensors(
        logprob=batch.batch["old_log_probs"],
        ref_logprob=batch.batch["ref_log_prob"],
        response_mask=batch.batch["response_mask"],
        windows=windows,
        estimators=estimators,
        prefix=prefix,
    )
