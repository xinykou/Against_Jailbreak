import argparse
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval_mmlu import eval_mmlu
from eval_triviaqa import eval_triviaqa
from eval_truthfulqa import eval_truthfulqa
import yaml
from safetensors import safe_open
import torch.nn as nn
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from soft_prompt_utils import process_soft_prompt_as_word_embedding

parser = argparse.ArgumentParser(description='evaluate generalization')

parser.add_argument('--model_name', type=str, default="")
parser.add_argument('--models_config_file', type=str, default="")
parser.add_argument('--eval_dataset_name', type=str, default="MMLU,TriviaQA, TruthfulQA")
parser.add_argument('--eval_dataset_dir', type=str, default="")
parser.add_argument('--output_result_dir', type=str, default="")
parser.add_argument("--soft_prompt_path", type=str, default=None,
                        help="The path to the soft prompt file")

args = parser.parse_args()
# Load model config file
config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
with open(config_file) as file:
    model_configs = yaml.full_load(file)

model_path = model_configs[args.model_name]['model']['model_name_or_path']

dataset_names = args.eval_dataset_name.split(',')
retain_datasets = []


# soft prompt addition
if args.soft_prompt_path is not None:
    use_soft_prompt = True
else:
    raise ValueError("Soft prompt path is required")

with safe_open(args.soft_prompt_path, framework='pt') as f:
    soft_prompt = f.get_tensor('soft_prompt')

for dataset_name in dataset_names:
    if dataset_name not in ['MMLU', 'BBH', 'TruthfulQA', 'TriviaQA', 'Fluency']:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    if dataset_name == 'MMLU':
        retain_datasets.append('MMLU')
        RETAIN_MMLU = 'mmlu.json'
        with open(os.path.join(args.eval_dataset_dir, RETAIN_MMLU), 'r') as f:
            retain_mmlu = json.load(f)
    if dataset_name == "TriviaQA":
        retain_datasets.append('TriviaQA')
        TRIVIAQA = 'triviaqa.json'
        with open(os.path.join(args.eval_dataset_dir, TRIVIAQA), 'r') as f:
            triviaqa = json.load(f)
    if dataset_name == "TruthfulQA":
        retain_datasets.append('TruthfulQA')
        TRUTHFULQA = 'truthfulqa.json'
        with open(os.path.join(args.eval_dataset_dir, TRUTHFULQA), 'r') as f:
            truthfulqa = json.load(f)

with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    e_tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    e_tokenizer.pad_token = e_tokenizer.eos_token

    e_tokenizer, new_input_embeddings = process_soft_prompt_as_word_embedding(model, e_tokenizer, soft_prompt)
    model.set_input_embeddings(new_input_embeddings.to(device=model.device, dtype=model.dtype))

    if 'MMLU' in retain_datasets:
        print("Evaluate mmlu...")
        eval_mmlu(model, e_tokenizer, retain_mmlu, batch_size=1,
                  output_result_dir=os.path.join(args.output_result_dir, f"{args.model_name}-MMLU-generalizability.json"),
                  use_prompt=False,
                  use_soft_prompt=use_soft_prompt,
                  soft_prompt=soft_prompt,
                  model_name=args.model_name
                  )

    if 'TriviaQA' in retain_datasets:
        print("Evaluate triviaqa...")
        eval_triviaqa(model, e_tokenizer, triviaqa, batch_size=16,
                      output_result_dir=os.path.join(args.output_result_dir, f"{args.model_name}-TriviaQA-generalizability.json"),
                      use_prompt=False,
                      use_soft_prompt=use_soft_prompt,
                      soft_prompt=soft_prompt,
                      model_name=args.model_name
                      )

    if 'TruthfulQA' in retain_datasets:
        print("Evaluate truthfulqa...")
        # truthfulqa = truthfulqa[:20]
        eval_truthfulqa(model, e_tokenizer, truthfulqa, batch_size=2,
                      output_result_dir=os.path.join(args.output_result_dir, f"{args.model_name}-TruthfulQA-generalizability.json"),
                      use_prompt=False,
                      use_soft_prompt=use_soft_prompt,
                      soft_prompt=soft_prompt,
                        model_name=args.model_name
                        )


