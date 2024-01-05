# inspired by https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py

# Adapted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import wandb
from datasets import load_dataset
from peft import (LoraConfig, TaskType, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling,
                          HfArgumentParser, Trainer, TrainingArguments)


@dataclass
class DataArguments:
    data_path: str


@dataclass
class ModelArguments:
    model_name_or_path: str
    trust_remote_code: bool = True


@dataclass
class TokenizerArguments:
    add_eos_token: bool = False
    add_bos_token: bool = False
    max_seq_length: int = 512
    padding_side: str = "right"


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_bias: str = "none"
    q_lora: bool = False


@dataclass
class WandbArguments:
    wandb_project: Optional[str] = None
    wandb_enable_checkpointing: bool = False
    wandb_resume_checkpoint: bool = False
    wandb_run_id: Optional[str] = None


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100 * trainable_params / all_param: .2f}%"
    )


def train():
    parser = HfArgumentParser(
        (
            DataArguments,
            ModelArguments,
            TokenizerArguments,
            TrainingArguments,
            LoraArguments,
            WandbArguments,
        )
    )

    (
        data_args,
        model_args,
        tokenizer_args,
        training_args,
        lora_args,
        wandb_args,
    ) = parser.parse_args_into_dataclasses()

    if training_args.report_to.lower() == "wandb" and not wandb_args.wandb_project:
        raise ValueError(
            "You must specify a wandb project name with --wandb_project when using wandb."
        )

    if wandb_args.wandb_resume_checkpoint and not wandb_args.wandb_run_id:
        raise ValueError(
            "You must specify a wandb run id with --wandb_run_id when resuming from a checkpoint."
        )
    
    if wandb_args.wandb_resume_checkpoint:
        run = wandb.init(
            project=wandb_args.wandb_project,
            id=wandb_args.wandb_run_id,
            resume="must",
        )
        artifact_dir = run.use_artifact(
            wandb_args.wandb_resume_checkpoint, type="model"
        ).download()

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if lora_args.q_lora
        else None,
        torch_dtype=compute_dtype,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )

    if lora_args.q_lora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side=tokenizer_args.padding_side,
        add_eos_token=tokenizer_args.add_eos_token,
        add_bos_token=tokenizer_args.add_bos_token,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_sample(sample):
        result = tokenizer(
            sample["text"],
            padding="max_length",
            max_length=training_args.max_seq_length,
            truncation=True,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = load_dataset(data_args.data_path).map(tokenize_sample)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    os.environ["WANDB_PROJECT"] = wandb_args.wandb_project
    if wandb_args.wandb_enable_checkpointing:
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    if wandb_args.wandb_resume_checkpoint:
        trainer.train(resume_from_checkpoint=artifact_dir)
    else:
        trainer.train()


if __name__ == "__main__":
    train()
