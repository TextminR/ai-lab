# inspired by https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py

# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team & Andreas Sünder. All rights reserved.
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

import logging
import os
from argparse import ArgumentParser
from typing import List

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType
from pydantic import BaseModel
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments,
                          DataCollatorForLanguageModeling)
from trl import SFTTrainer, is_xpu_available

tqdm.pandas()


# Define and parse arguments.
class ScriptArguments(BaseModel):

  job_name: str
    
  model_name: str
  trust_remote_code: bool = True
  tokenizer_kwargs: dict = None
  tokenizer_use_eos_token: bool = False
  dataset_prompt_template: str = None
  dataset_name: str
  dataset_context_col: str
  dataset_instruction_col: str
  dataset_response_col: str
  seq_length: int

  use_peft: bool = False
  load_in_4bit: bool = False

  lora_r: int = 64
  lora_alpha: int = 16
  lora_dropout: float = 0.05
  target_modules: List[str]

  output_dir: str = './output'
  logging_dir: str = './logs'
  num_train_epochs: int = 1
  warmup_steps: int = 0
  max_steps: int = -1
  learning_rate: float = 1e-3
  batch_size: int = 8
  gradient_checkpointing: bool = False
  gradient_accumulation_steps: int = 1
  optim: str
  logging_steps: int = 10
  logging_strategy: str = 'steps'
  do_eval: bool = False
  eval_strategy: str = 'no'
  eval_steps: int = 0
  save_strategy: str = 'no'
  save_steps: int = 0
  save_total_limit: int = 5
  use_data_collator: bool = True

  report_to_wandb: bool = False
  wandb_project: str = None

  push_to_hub: bool = False
  hub_model_id: str = None


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)

  parser = ArgumentParser()
  parser.add_argument('config_path', type=str, help='Path to the training config file')
  
  if not os.path.exists(parser.parse_args().config_path):
    raise ValueError('Config file does not exist.')

  args = ScriptArguments.model_validate_json(open(parser.parse_args().config_path, 'r').read())
  
  if args.report_to_wandb and not args.wandb_project:
    raise ValueError('If reporting to wandb, please specify a project name.')
  if args.report_to_wandb:
    os.environ['WANDB_PROJECT'] = args.wandb_project

  if args.dataset_prompt_template is None:
    logging.warning('No prompt template specified. Using default.')
    prompt_template = '### Context: {} ### Instruction: {} ### Response: {}'
  else:
    prompt_template = args.dataset_prompt_template

  if args.push_to_hub and not args.hub_model_id:
    raise ValueError('If pushing to hub, please specify a model id.')

  quantization_config = None
  device_map = None 

  if args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # Copy the model to each device
    device_map = (
      {"": f"xpu:{Accelerator().local_process_index}"}
      if is_xpu_available()
      else {"": Accelerator().local_process_index}
    )

  model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    quantization_config=quantization_config,
    trust_remote_code=args.trust_remote_code,
    torch_dtype=torch.float16,
    device_map='auto',
  )
  
  train_dataset = load_dataset(args.dataset_name, split='train')
  eval_dataset = load_dataset(args.dataset_name, split='validation') if args.do_eval else None
  
  training_args = TrainingArguments(
    output_dir=args.output_dir,
    logging_dir=args.logging_dir,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    learning_rate=args.learning_rate,
    warmup_steps=args.warmup_steps,
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    bf16=True,
    optim=args.optim,
    logging_strategy=args.logging_strategy,
    logging_steps=args.logging_steps,
    do_eval=args.do_eval,
    evaluation_strategy=args.eval_strategy,
    eval_steps=args.eval_steps,
    save_strategy=args.save_strategy,
    save_steps=args.save_steps,
    save_total_limit=args.save_total_limit,
    report_to='wandb' if args.report_to_wandb else None,
    run_name=args.job_name,
    push_to_hub=args.push_to_hub,
    hub_model_id=args.hub_model_id,
  )
  
  peft_config = None
  if args.use_peft:
    peft_config = LoraConfig(
      r=args.lora_r,
      lora_alpha=args.lora_alpha,
      target_modules=args.target_modules,
      bias='none',
      lora_dropout=args.lora_dropout,
      task_type=TaskType.CAUSAL_LM,
    )
  
  tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, **args.tokenizer_kwargs)
  if args.tokenizer_use_eos_token:
    tokenizer.pad_token = tokenizer.eos_token

  def formatting_prompts_func(examples):
    output_text = []
    
    for i in range(len(examples[args.dataset_instruction_col])):
      output_text.append(prompt_template.format(
        examples[args.dataset_context_col][i],
        examples[args.dataset_instruction_col][i],
        examples[args.dataset_response_col][i],
      ))

    return output_text

  trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_prompts_func,
    max_seq_length=args.seq_length,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
      if args.use_data_collator else None,
  )

  trainer.train()