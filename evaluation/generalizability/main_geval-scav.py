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

parser = argparse.ArgumentParser(description='evaluate generalization')

parser.add_argument('--model_name', type=str, default="")
parser.add_argument('--models_config_file', type=str, default="")
parser.add_argument('--eval_dataset_name', type=str, default="MMLU,TriviaQA, TruthfulQA")
parser.add_argument('--eval_dataset_dir', type=str, default="")
parser.add_argument('--output_result_dir', type=str, default="")
parser.add_argument("--selected_layer_ids", type=str, default="",
                        help="Selected layer ids for vLLM generation")
parser.add_argument("--save_mask_path", type=str, default=None,
                        help="The path for saving layer mask positions")
parser.add_argument("--ablation_coefficient", type=float, default=1.0,
                    help="The coefficient for ablation for scav, default is 1.0")

args = parser.parse_args()
# Load model config file
config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
with open(config_file) as file:
    all_configs = yaml.full_load(file)

model_path = all_configs[args.model_name]['model']['model_name_or_path']

dataset_names = args.eval_dataset_name.split(',')
retain_datasets = []

use_masked_scav = False
use_ablation_coefficient = False
map_current_direction_mask_dict = {}
if args.save_mask_path is not None:
    masked_scav_dict = safe_open(f'{args.save_mask_path}',
                                 framework='pt', device=0)
    refusal_direction_mask_index = masked_scav_dict.get_tensor('mask_index')  # n_layer, dim
    selected_layer_ids = args.selected_layer_ids.split(",")
    selected_layer_ids = [int(i) for i in selected_layer_ids]
    print(f"Selected layer ids: {selected_layer_ids}")
    assert len(selected_layer_ids) > 1, "Please provide more than one layer id for scav"
    assert len(selected_layer_ids) == refusal_direction_mask_index.size(0), "The number of selected layer ids should be equal to the number of layers in the model"
    for inds in range(len(selected_layer_ids)):
        map_current_direction_mask_dict[selected_layer_ids[inds]] = refusal_direction_mask_index[inds]
    use_masked_scav = True

if args.ablation_coefficient != 1.0:
    use_ablation_coefficient = True

if args.selected_layer_ids == "all":
    mask_layer_ids = []
else:
    selected_layer_ids = args.selected_layer_ids.split(",")
    selected_layer_ids = [int(i) for i in selected_layer_ids]
    if args.model_name == "llama3_8B" or args.model_name == "mistral_7B":
        layer_nums = 32
        mask_layer_ids = []
        for i in range(layer_nums):
            if i not in selected_layer_ids:
                mask_layer_ids.append(i)
    elif args.model_name == "gemma_7B":
        layer_nums = 28
        mask_layer_ids = []
        for i in range(layer_nums):
            if i not in selected_layer_ids:
                mask_layer_ids.append(i)
    else:
        raise ValueError(f"Model {args.model_name} not supported")


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
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    from scav.perturbation import Perturbation
    from scav.model_generation import ModelGeneration

    classifier_path = all_configs[args.model_name].get('classifier_path', None)
    clfr = torch.load(classifier_path)
    pert = Perturbation(clfr, target_probability=0.05,
                        layer_mask=mask_layer_ids,
                        use_masked_scav=use_masked_scav,
                        map_current_direction_mask_dict=map_current_direction_mask_dict,
                        use_ablation_coefficient=use_ablation_coefficient,
                        ablation_coefficient=args.ablation_coefficient
                        )
    llm_gen = ModelGeneration(args.model_name)
    llm_gen.set_perturbation(pert)
    model = llm_gen.model
    model.eval()
    e_tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
    e_tokenizer.pad_token = e_tokenizer.eos_token

    os.makedirs(args.output_result_dir, exist_ok=True)

    if 'MMLU' in retain_datasets:
        print("Evaluate mmlu...")
        eval_mmlu(model, e_tokenizer, retain_mmlu, batch_size=1,
                  output_result_dir=os.path.join(args.output_result_dir, f"{args.model_name}-MMLU-generalizability.json"),
                  use_prompt=False,
                  model_name=args.model_name)

    if 'TriviaQA' in retain_datasets:
        print("Evaluate triviaqa...")
        eval_triviaqa(model, e_tokenizer, triviaqa, batch_size=16,
                      output_result_dir=os.path.join(args.output_result_dir, f"{args.model_name}-TriviaQA-generalizability.json"),
                      use_prompt=False,
                      model_name=args.model_name)

    if 'TruthfulQA' in retain_datasets:
        print("Evaluate truthfulqa...")
        # truthfulqa = truthfulqa[:20]
        eval_truthfulqa(model, e_tokenizer, truthfulqa, batch_size=2,
                      output_result_dir=os.path.join(args.output_result_dir, f"{args.model_name}-TruthfulQA-generalizability.json"),
                      use_prompt=False,
                      model_name=args.model_name)
