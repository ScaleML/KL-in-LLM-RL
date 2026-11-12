from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop


class CapacityProbe(SingleTurnAgentLoop):
    """Fill the response budget to exercise rollout and FSDP at the configured limit."""

    def __init__(self, *args, name=None, **kwargs):
        super().__init__(*args, **kwargs)

    async def run(self, sampling_params, **kwargs):
        params = {
            **sampling_params,
            "max_tokens": self.response_length,
            "min_tokens": self.response_length,
            "ignore_eos": True,
        }
        output = await super().run(params, **kwargs)
        if len(output.response_ids) != self.response_length:
            raise RuntimeError(f"Capacity probe got {len(output.response_ids)} of {self.response_length} response tokens")
        return output
