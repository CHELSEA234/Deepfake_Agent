# Aggregator — LLaVA-1.5-7B GRPO

Requires an upstream LLaVA checkout on `PYTHONPATH`.

```bash
pip install -r requirements.txt
```

```bash
export UPSTREAM_LLAVA_ROOT=/path/to/LLaVA
export MODEL_NAME_OR_PATH=llava-hf/llava-1.5-7b-hf
export DATA_PATH=/path/to/train.json
export IMAGE_FOLDER=/path/to/images
export OUTPUT_DIR=/path/to/output
export DOMAIN_METHOD_MAP="face=2,editing=1,semantic=3"
./train_aggregator.sh
```
