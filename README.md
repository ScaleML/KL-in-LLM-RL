# Future-KL Regularized GRPO: Process-Level Credit Assignment from $f$-Divergence Regularization
This is the codebase for paper "Future-KL Regularized GRPO: Process-Level Credit Assignment from $f$-Divergence Regularization".

## Data Preparation
Training data: [FlippyDora/dapo-filter](https://huggingface.co/datasets/FlippyDora/dapo-filter)

Evaluation benchmarks:
1. [CorrectKLinRL/olympiadbench](https://huggingface.co/datasets/CorrectKLinRL/olympiadbench)
2. [CorrectKLinRL/minerva_math](https://huggingface.co/datasets/CorrectKLinRL/minerva_math)
3. [CorrectKLinRL/amc23](https://huggingface.co/datasets/CorrectKLinRL/amc23)
4. [CorrectKLinRL/aime25](https://huggingface.co/datasets/CorrectKLinRL/aime25)
5. [CorrectKLinRL/aime24](https://huggingface.co/datasets/CorrectKLinRL/aime24)
6. [CorrectKLinRL/math500](https://huggingface.co/datasets/CorrectKLinRL/math500)

## Training
Use the scripts under `verl/rl-kl`.
```bash
cd verl
bash rl-kl/train.sh
```

## Citation
```
@article{yao2026prl,
  title={PRL: Process Reward Learning Improves LLMs' Reasoning Ability and Broadens the Reasoning Boundary},
  author={Yao, Jiarui and Wang, Ruida and Zhang, Tong},
  journal={arXiv preprint arXiv:2601.10201},
  year={2026}
}
```