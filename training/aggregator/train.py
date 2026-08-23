#!/usr/bin/env python3
"""Historical multimodal GRPO entry point for the LLaVA-1.5-7B forgery aggregator.

The aggregator consumes three expert reports per image and emits a four-tag
verdict (`<think>`, `<answer>`, `<method>`, `<score>`) scored by five reward
functions. This thin snapshot requires an importable upstream LLaVA checkout in
addition to its pinned Hugging Face/TRL dependencies.
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import transformers
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextProcessor
from trl import GRPOConfig

from llava_grpotrainer import LLaVAGRPOTrainer


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Hugging Face model ID or local LLaVA-1.5 7B path."}
    )
    trust_remote_code: bool = True
    attention_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "Attention backend; use 'none' to omit the setting."},
    )


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Training JSON path."})
    image_folder: str = field(metadata={"help": "Root for image paths in JSON."})
    domain_method_map: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "Repeated or comma-separated SUBSTRING=METHOD_NUMBER pairs used by "
                "tool_selection_reward, e.g. 'face=2,editing=1,semantic=3'. Matching is "
                "case-insensitive and the first matching pair wins. Records that match "
                "nothing score 0.0 for that reward."
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = None
    optim: str = "adamw_torch"
    remove_unused_columns: bool = False
    model_max_length: int = 512
    max_completion_length: Optional[int] = None
    num_generations: int = 2
    beta: float = 0.4
    max_grad_norm: float = 1.0
    reward_weights: Optional[list[float]] = None
    group_by_modality_length: bool = False
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    bits: int = 16
    double_quant: bool = True
    quant_type: str = "nf4"


class GRPODataset(Dataset):
    """Keep the historical LLaVA conversation JSON unprocessed until collation."""

    def __init__(self, data_path: str, image_folder: str) -> None:
        with open(data_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError("Training JSON must contain a top-level list.")
        self.records = records
        self.image_folder = image_folder

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        # The trainer expects each item wrapped in a single-element list.
        return [self.records[index]]

    @property
    def lengths(self) -> list[int]:
        length_list = []
        for sample in self.records:
            img_tokens = 128 if sample.get("image") else 0
            conversations = sample.get("conversations", [])
            length_list.append(
                sum(len(turn.get("value", "").split()) for turn in conversations) + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self) -> list[int]:
        lengths = []
        for sample in self.records:
            conversations = sample.get("conversations", [])
            length = sum(len(turn.get("value", "").split()) for turn in conversations)
            length = max(length, 1)
            lengths.append(length if sample.get("image") else -length)
        return lengths


_ORDERED_RE = re.compile(
    r'^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*<method>.*?</method>\s*<score>.*?</score>\s*$',
    re.DOTALL
)

_TAGS = ("think", "answer", "method", "score")


def format_reward(completions, **kwargs):
    """Reward the ordered <think>/<answer>/<method>/<score> envelope."""
    rewards = []
    for content in completions:
        s = content or ""
        if not _ORDERED_RE.match(s):
            rewards.append(0.0)
            continue
        # ensure exactly one opening & one closing tag for each
        ok = all(
            len(re.findall(fr'<{t}>', s)) == 1 and len(re.findall(fr'</{t}>', s)) == 1
            for t in _TAGS
        )
        rewards.append(1.0 if ok else 0.0)
    return rewards


def accuracy_reward(answers, completions, **kwargs):
    """Reward a <answer> verdict that agrees with the real/fake label."""
    reward_lst = []
    for i in range(len(completions)):
        match = re.search(r"<answer>(.*?)</answer>", completions[i], re.DOTALL | re.IGNORECASE)
        pred = match.group(1).strip() if match else ""
        pred = pred.split('.')[0].lower()
        answer = answers[i].lower()
        if 'real' in pred or 'captured by camera' in pred or 'not manipulated' in pred or 'not generated' in pred:
            if 'real' in answer:
                reward_lst.append(1.0)
            else:
                reward_lst.append(0.0)
        elif 'fake' in pred or 'manipulated' in pred or 'ai-generated' in pred:
            if 'fake' in answer or 'manipulated' in answer or 'ai-generated' in answer:
                reward_lst.append(1.0)
            else:
                reward_lst.append(0.0)
        else:
            reward_lst.append(0.0)
    return reward_lst


def tool_selection_reward(method_names, completions, **kwargs):
    """Reward a <method> choice that matches the expert expected for the image."""
    reward_lst = []
    for i in range(len(completions)):
        match = re.search(r"<method>(.*?)</method>", completions[i], re.DOTALL | re.IGNORECASE)
        pred = match.group(1).strip() if match else ""
        pred = pred.lower()
        method_name = method_names[i].lower()
        if 'method 1' in pred or 'method one' in pred:
            if '1' in method_name:
                reward_lst.append(1.0)
            else:
                reward_lst.append(0.0)
        elif 'method 2' in pred or 'method two' in pred:
            if '2' in method_name:
                reward_lst.append(1.0)
            else:
                reward_lst.append(0.0)
        elif 'method 3' in pred or 'method three' in pred:
            if '3' in method_name:
                reward_lst.append(1.0)
            else:
                reward_lst.append(0.0)
        else:
            reward_lst.append(0.0)
    return reward_lst


def reasoning_length_reward(completions, **kwargs):
    """
    Reward function to encourage lengthy reasoning (0 to 1 scale)
    """
    rewards = []

    for response in completions:
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if not think_match:
            rewards.append(0.0)
            continue

        reasoning = think_match.group(1).strip()
        word_count = len(reasoning.split())

        # Progressive scaling
        if word_count < 30:
            reward = word_count / 120.0  # 0-0.25 for <30 words
        elif word_count < 100:
            reward = 0.25 + (word_count - 30) / 280.0  # 0.25-0.5 for 30-100 words
        elif word_count < 150:
            reward = 0.5 + (word_count - 100) / 200.0  # 0.5-1.0 for 100-150 words
        else:
            reward = 1.0

        rewards.append(reward)

    return rewards


def strict_reasoning_format_reward(completions, **kwargs):
    """
    Strict reward function requiring explicit section headers
    Takes a list of responses and returns a list of rewards
    """
    rewards = []

    for response in completions:
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if not think_match:
            rewards.append(0.0)
            continue

        reasoning = think_match.group(1).strip()

        # Required section patterns
        sections = [
            r"\d+\.\s*IMAGE CONTENT",
            r"\d+\.\s*METHOD ANALYSIS",
            r"\d+\.\s*CREDIBILITY ASSESSMENT",
            r"\d+\.\s*FINAL DECISION"
        ]

        # Check each section
        section_count = 0
        for pattern in sections:
            if re.search(pattern, reasoning, re.IGNORECASE):
                section_count += 1

        # Base score from sections (0.7 max)
        base_score = (section_count / 4.0) * 0.7

        # Method coverage bonus
        methods_covered = 0
        for i in range(1, 4):
            if re.search(f"method {i}", reasoning, re.IGNORECASE):
                methods_covered += 1

        method_bonus = (methods_covered / 3.0) * 0.3

        rewards.append(min(1.0, base_score + method_bonus))

    return rewards


REWARD_FUNCS = [
    format_reward,
    accuracy_reward,
    tool_selection_reward,
    reasoning_length_reward,
    strict_reasoning_format_reward,
]


def parse_domain_method_map(entries: Optional[list[str]]) -> dict[str, str]:
    """Parse repeated or comma-separated SUBSTRING=METHOD_NUMBER pairs."""
    mapping: dict[str, str] = {}
    for entry in entries or []:
        for pair in str(entry).split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Expected SUBSTRING=METHOD_NUMBER in --domain_method_map, got {pair!r}."
                )
            substring, method = pair.split("=", 1)
            substring, method = substring.strip().lower(), method.strip()
            if not substring or not method:
                raise ValueError(
                    f"Both sides of a --domain_method_map pair must be non-empty, got {pair!r}."
                )
            mapping[substring] = method
    return mapping


def make_quantization_config(args: TrainingArguments) -> Optional[BitsAndBytesConfig]:
    """Return a bitsandbytes config, or None at the default 16-bit setting."""
    if args.bits not in (4, 8):
        return None
    compute_dtype = (
        torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32
    )
    return BitsAndBytesConfig(
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        llm_int8_skip_modules=["mm_projector"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.double_quant,
        bnb_4bit_quant_type=args.quant_type,
    )


def make_grpo_config(args: TrainingArguments) -> GRPOConfig:
    config = GRPOConfig(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        tf32=args.tf32,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_on_each_node=False,
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        max_prompt_length=args.model_max_length,
        max_completion_length=args.max_completion_length or args.model_max_length,
        optim=args.optim,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        reward_weights=args.reward_weights,
    )
    config.cache_dir = args.cache_dir
    config.group_by_modality_length = args.group_by_modality_length
    return config


def main() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dataset = GRPODataset(data_args.data_path, data_args.image_folder)
    domain_method_map = parse_domain_method_map(data_args.domain_method_map)

    if "llava-v1.6" in model_args.model_name_or_path:
        processor = LlavaNextProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=training_args.cache_dir,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            cache_dir=training_args.cache_dir,
        )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    peft_config = None  # Historical script left this undefined when LoRA was off.
    if training_args.lora_enable:
        peft_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=None,  # resolved by the trainer
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )

    model_init_kwargs: dict[str, Any] = {
        "trust_remote_code": model_args.trust_remote_code
    }
    if model_args.attention_implementation not in (None, "", "none"):
        model_init_kwargs["attn_implementation"] = model_args.attention_implementation

    grpo_args = make_grpo_config(training_args)
    grpo_args.model_init_kwargs = model_init_kwargs
    trainer = LLaVAGRPOTrainer(
        model=model_args.model_name_or_path,
        processing_class=processor,
        reward_funcs=REWARD_FUNCS,
        args=grpo_args,
        peft_config=peft_config,
        bnb_configs=make_quantization_config(training_args),
        train_dataset=dataset,
        image_folder=data_args.image_folder,
        domain_method_map=domain_method_map,
    )

    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if checkpoints else None)
    if trainer.accelerator.is_main_process:
        final_dir = pathlib.Path(training_args.output_dir) / "final"
        trainer.save_model(str(final_dir))
        processor.save_pretrained(str(final_dir))


if __name__ == "__main__":
    main()
