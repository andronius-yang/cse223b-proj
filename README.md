# cse223b-proj

Course project repository. Current tool:

- `traffic-gen/`: trace-driven MoE expert-selection traffic matrix generator.

Run it from its own folder:

```bash
cd traffic-gen
source .venv/bin/activate
python3 generate.py
```

## Credits
This project uses a local partial subset of the MoE expert-selection traces from Yu et al., "Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference," arXiv:2510.05497, and the associated Hugging Face dataset `core12345/MoE_expert_selection_trace`. The subset used here is from the Llama-4-Maverick MMLU traces.
* [arXiv paper](https://arxiv.org/abs/2510.05497)
* [Hugging Face dataset](https://huggingface.co/datasets/core12345/MoE_expert_selection_trace)