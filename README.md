# Deepfake_Agent

The code for: Deepfake-Agent: Aggregating Semantic Forgery Clues for Generalizable Detection

Aggregation of complementary forensic reports for general image-forgery
detection. 

- image editing — [FakeShield](https://github.com/zhipeixu/FakeShield)
- face swap — [M2F2-Det](https://github.com/CHELSEA234/M2F2_Det)
- semantic — GPT-4o / Gemini proposers with a LLaVA-1.6-34B adjudicator
- aggregator — LLaVA-1.5-7B trained with GRPO

Upstream expert repositories are not vendored here. Install them separately and
follow their licenses.

```text
experts/              semantic, fakeshield, m2f2 adapters
dataset_construction/ expert-report merge CLI
training/             semantic_expert and aggregator GRPO snapshots
```

## 1. Experts and semantic adjudicator

Each expert then runs from its own directory against upstream
weights, and its output is normalized to a shared schema before aggregation.

The GPT-4o and Gemini proposers produce paired reports that are converted into
LLaVA adjudicator data, which trains the LLaVA-1.6-34B semantic expert.

```bash
cd training/semantic_expert
export UPSTREAM_LLAVA_ROOT=/path/to/LLaVA
export MODEL_NAME_OR_PATH=llava-hf/llava-v1.6-34b-hf
export DATA_PATH=/path/to/semantic.json
export IMAGE_FOLDER=/path/to/images
export OUTPUT_DIR=/path/to/output
./train_semantic_expert.sh
```

## 2. Aggregator

Trains on the merged three-expert dataset. Responses use a four-tag envelope
scored by five reward terms.

```bash
cd training/aggregator
export UPSTREAM_LLAVA_ROOT=/path/to/LLaVA
export MODEL_NAME_OR_PATH=llava-hf/llava-1.5-7b-hf
export DATA_PATH=/tmp/train.json
export IMAGE_FOLDER=/path/to/images
export OUTPUT_DIR=/path/to/output
export DOMAIN_METHOD_MAP="face=2,editing=1,semantic=3"
./train_aggregator.sh
```

Merge a resulting adapter:

```bash
python ../semantic_expert/merge_lora_weights.py --base llava-hf/llava-1.5-7b-hf \
  --lora /path/to/adapter --out /path/to/merged
```

## Notes
Expert outputs may be subject to source-dataset and API-provider terms. This is
research code and requires external repositories and weights to run end to end.
