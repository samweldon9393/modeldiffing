# MiniReason
## Installation

To set up your environment and install necessary dependencies, follow these steps:

```bash
# Upgrade pip
pip install --upgrade pip

# Install uv for managing the virtual environment and dependencies
pip install uv

# Create a virtual environment using uv
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install sglang kernel
uv pip install sgl-kernel --force-reinstall --no-deps

# Install project dependencies from requirements.txt
uv pip install -r requirements.txt

# Install sglang with all optional dependencies, using provided wheels for CUDA 12.4 and PyTorch 2.4
uv pip install "sglang[all]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer
```

## Metrics

- **pass@1:**
  - Only one generated answer is allowed for each question.
  - The model passes if the single generated answer is correct.

- **match@8:**
  - Eight answers are sampled.
  - The model passes if the majority (at least 5 out of 8) are correct.

- **pass@8:**
  - Eight answers are sampled.
  - The model passes if at least one of the eight generated answers is correct.

