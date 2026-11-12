"""Score the answer content of single-turn Gemma math responses."""

import re

from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.utils.reward_score import math_verify


THOUGHT_START = "<|channel>thought\n"
THOUGHT_END = "<channel|>"
MODEL_TURN_START = "<|turn>model\n"
STOP_MARKERS = re.compile(r"<turn\|>|<eos>|<\|tool_(?:call|response)>")


def final_answer(text):
    """Match the text-only portion of Gemma's response schema."""
    content = text.strip().removeprefix(MODEL_TURN_START)
    content = STOP_MARKERS.split(content, maxsplit=1)[0].strip()
    if content.startswith(THOUGHT_START):
        _, thought_end, content = content.partition(THOUGHT_END)
        if not thought_end:
            return ""
    if "<|channel>" in content or THOUGHT_END in content or "<|turn>" in content:
        return ""
    return content.strip()


def score_response(text, ground_truth):
    answer = final_answer(text)
    score = math_verify.compute_score(answer, ground_truth) if answer else 0.0
    return {"score": float(score), "acc": float(score), "has_final": float(bool(answer))}


def warm_verifier():
    # Worker startup time counts against the verifier's timeout.
    if math_verify.compute_score(r"\boxed{1}", "1", timeout=90.0) != 1.0:
        raise RuntimeError("Math-Verify warmup did not recognize a known correct answer")


class GemmaMathRewardManager(DAPORewardManager):
    """Reuse DAPO's configuration and decode raw Gemma response tokens for math scoring."""

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score, **kwargs)
        warm_verifier()

    async def run_single(self, data):
        if len(data) != 1:
            raise ValueError("Expected one rollout response per reward call")
        item = data[0]
        response_ids = item.batch["responses"]
        response_length = int(item.batch["attention_mask"][-response_ids.numel() :].sum().item())
        valid_ids = response_ids[:response_length].tolist()
        ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]

        def score():
            raw_text = self.tokenizer.decode(valid_ids, skip_special_tokens=False)
            return score_response(raw_text, ground_truth)

        info = await self.loop.run_in_executor(None, score)
        reward = info["score"]
        buffer = self.overlong_buffer_cfg
        if buffer is not None and buffer.enable:
            excess = response_length - (self.max_resp_len - buffer.len)
            overlong_reward = min(-excess / buffer.len * buffer.penalty_factor, 0.0)
            reward += overlong_reward
            if buffer.log:
                info["overlong_reward"] = overlong_reward
                info["overlong"] = overlong_reward < 0
        return {"reward_score": reward, "reward_extra_info": info}
