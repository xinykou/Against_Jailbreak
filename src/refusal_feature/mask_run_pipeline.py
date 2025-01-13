import torch
import random
import json
import os

os.environ["OPENAI_API_KEY"] = "sk-"
import sys
from safetensors.torch import save_file

print("sys.path:", sys.path)
from submodules.generate_directions import generate_masks
from utilss.hook_utils import get_all_direction_ablation_hooks, add_hooks



def run_pipeline(step=1,
                 harmful_data=None,
                 harmless_data=None,
                 model_base=None,
                 args=None,
                 selected_layer_ids=None
                 ):
    # Load and sample datasets
    random.Random(step)
    max_sample = 32
    harmful_train = random.sample(harmful_data, max_sample)
    harmless_train = random.sample(harmless_data, max_sample)

    # 1. Generate candidate refusal directions: n_layers * d_model
    candidate_masks, mean_last_dim, variance_last_dim, mean_harmless_last_dim = generate_masks(model_base,
                                                                                               harmful_train,
                                                                                               harmless_train,
                                                                                               selected_layer_ids,
                                                                                               top_k=args.retain_topk,
                                                                                               feature_type=args.refusal_feature_type
                                                                                              )
    # save mask positions
    try:
        if args.save_mask_path is not None:
            os.makedirs(args.save_mask_path, exist_ok=True)
            cpu_mask_index = candidate_masks.cpu()
            mask_index_dict = {"mask_index": cpu_mask_index}  # layer, d_model
            save_file(mask_index_dict, os.path.join(args.save_mask_path, "mask_index.safetensors"))
            print(f"-------Mask index saved to {args.save_mask_path}-------------------")

    except Exception as e:
        print(f"Error in saving mask index: {e}")
        pass

    mask_index_device = candidate_masks.to(model_base.model.device)
    directions_device = mean_last_dim.to(model_base.model.device)
    if variance_last_dim is not None and mean_harmless_last_dim is not None:
        variance_last_dim = variance_last_dim.to(model_base.model.device)
        harmless_directions_device = mean_harmless_last_dim.to(model_base.model.device)
    else:
        variance_last_dim = None
        harmless_directions_device = None

    return mask_index_device, directions_device, variance_last_dim, harmless_directions_device



from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os
from tqdm import tqdm
import argparse
import numpy as np
if __name__ == "__main__":
    # os.environ[("CUDA_VISIBLE_DEVICES")] = "5"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model',
                        default='')
    parser.add_argument('--model_path', type=str, required=False, help='Path to the model',
                        default='/media/5/yx/model_cache/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--test_dataset_dir_path', type=str, required=False, help='Path to the test dataset directory',
                        default='/media/5/yx/against_jailbreak/data/build_vector/harmful_val.json')
    parser.add_argument("--output_dir", type=str, required=False, help="Output directory for the pipeline",
                        default="/media/5/yx/against_jailbreak/results/vanilla/Self_mask_double_ReFAT_Attack/llama3_8B-HarmBench-top_10-delta_0.1-safety")
    parser.add_argument("--selected_layer_ids", type=str, default="8-32")
    parser.add_argument("--retain_topk", type=int, default=10)
    parser.add_argument("--deltas", type=str, help="The threshold coefficient for the refusal feature",
                        default="0.1")
    parser.add_argument("--target_model_name", type=str, default="llama3_8B")
    parser.add_argument("--dataset_name", type=str, default="HarmBench")
    parser.add_argument("--beta", type=float, default=0.1, help="The beta value for the coefficient of perturbation")
    parser.add_argument("--refusal_feature_type", type=str, default="constant", choices=["constant", "variable", "mask_and_map"],
                        help="The type of refusal feature")
    parser.add_argument("--save_mask_path", type=str, default=None,
                        help="The path to save the mask positions")
    args = parser.parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    config = base_model.config
    n_layers = config.num_hidden_layers

    if args.model_name == "llama3_8B":
        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}<|eot_id|>"
        user_tag = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    elif args.model_name == "mistral_7B":
        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}</s>"
        user_tag = "<s> [INST]"
        assistant_tag = "[/INST]"

    elif args.model_name == "gemma_7B":
        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
        user_tag = "<bos><start_of_turn>user\n"
        assistant_tag = "<|end_of_turn|>\n<start_of_turn>model\n"
    elif args.model_name == "qwen2_7B":
        one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
        user_tag = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        assistant_tag = "<|im_end|>\n<|im_start|>assistant\n"

    with open("data/build_vector/harmful_train.json") as file:
        harmful_dataset = json.load(file)
        harmful_nums = len(harmful_dataset)
    with open("data/build_vector/harmless_train.json") as file:
        harmless_dataset = json.load(file)
        harmless_nums = len(harmless_dataset)

    with open(args.test_dataset_dir_path) as file:
        test_dataset = json.load(file)

    harmful_train, harmless_train, test_data = [], [], []

    for i, d in enumerate(harmful_dataset):
        formatted_input = one_shot_template.format(
            user_tag=user_tag, assistant_tag=assistant_tag,
            instruction=d['instruction'], response="").split("<SEPARATOR>")[0]

        harmful_train.append(formatted_input)

    for i, d in enumerate(harmless_dataset):
        formatted_input = one_shot_template.format(
            user_tag=user_tag, assistant_tag=assistant_tag,
            instruction=d['instruction'], response="").split("<SEPARATOR>")[0]

        harmless_train.append(formatted_input)

    print("harmful_dataset[0]:", harmful_train[0])
    print("harmless_dataset[0]:", harmless_train[0])
    print("harmful_nums/harmless_nums:", harmful_nums, harmless_nums)

    max_num = 100 if len(test_dataset) > 100 else len(test_dataset)
    random.seed(0)
    test_dataset = random.sample(test_dataset, max_num)
    for i, d in enumerate(test_dataset):
        if "prompt" in d:
            input = d['prompt']
            formatted_input = one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=input, response="").split("<SEPARATOR>")[0]
        elif "instruction" in d:
            input = d['instruction']
            formatted_input = one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=input, response="").split("<SEPARATOR>")[0]

        test_data.append(formatted_input)

    feature_tokenized_kwargs = dict(max_length=64, padding='max_length', truncation=True, return_tensors="pt",
                                    add_special_tokens=False)
    harmful_data_tokenized = [tokenizer(one_harmful_instance, **feature_tokenized_kwargs) for
                                   one_harmful_instance in harmful_train]
    harmless_data_tokenized = [tokenizer(one_harmless_instance, **feature_tokenized_kwargs) for
                                    one_harmless_instance in harmless_train]

    test_dataset_tokenized = [tokenizer(one_test_instance, **feature_tokenized_kwargs) for
                                  one_test_instance in test_data]

    if "all" in args.selected_layer_ids:
        select_layer_ids = list(range(n_layers))
    elif "-" in args.selected_layer_ids:
        id_range = [int(i) for i in args.selected_layer_ids.split("-")]
        selected_layer_ids = list(range(id_range[0], id_range[1]))
    elif "," in args.selected_layer_ids:
        selected_layer_ids = [int(i) for i in args.selected_layer_ids.split(",")]

    mask_index_device, directions_device, variance_last_dim, harmless_directions_device = \
                                                                        run_pipeline(step=0,
                                                                                      harmful_data=harmful_data_tokenized,
                                                                                      harmless_data=harmless_data_tokenized,
                                                                                      model_base=base_model,
                                                                                      args=args,
                                                                                      selected_layer_ids=selected_layer_ids
                                                                                      )


    mask_direction_device = mask_index_device * directions_device
    if args.refusal_feature_type == "variable":
        mask_harmless_direction_device = mask_index_device * harmless_directions_device
        mask_variance_device = mask_index_device * variance_last_dim
    else:
        mask_harmless_direction_device = None
        mask_variance_device = None

    delta_all = [float(delta) for delta in args.deltas.split(",")]
    delta_all = [round(delta, 2) for delta in delta_all]
    for delta in delta_all:
        delta_device = torch.tensor(delta, device=base_model.model.device)
        belta_device = torch.tensor(args.beta, device=base_model.model.device)
        ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(base_model,
                                                                                      mask_index_device,
                                                                                      mask_direction_device,
                                                                                      delta_device,
                                                                                      mask_variance_device,
                                                                                      mask_harmless_direction_device,
                                                                                      pert_coef=belta_device,
                                                                                      selected_layer_ids=selected_layer_ids,
                                                                                      refusal_feature_type=args.refusal_feature_type
                                                                                      )

        generation_config = GenerationConfig(max_new_tokens=512, do_sample=False)
        batch_size = 8
        completions = []
        # test_dataset_tokenized = test_dataset_tokenized[:8]
        test_nums = len(test_dataset_tokenized)
        for i in tqdm(range(0, test_nums, batch_size)):
            tokenized_instructions = test_dataset_tokenized[i:i + batch_size]
            input_ids = torch.cat([inp["input_ids"].to(base_model.device) for inp in tokenized_instructions], dim=0)
            attention_mask = torch.cat([inp["attention_mask"].to(base_model.device) for inp in tokenized_instructions], dim=0)
            with add_hooks(module_forward_pre_hooks=ablation_fwd_pre_hooks,
                           module_forward_hooks=ablation_fwd_hooks,
                           sample_id=i), torch.no_grad():
                generation_toks = base_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[:, input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append({
                        'instruction': test_data[i + generation_idx],
                        'category': test_dataset[i + generation_idx]['category'] if "category" in test_dataset[i + generation_idx] else None,
                        'output': tokenizer.decode(generation, skip_special_tokens=True).strip()
                    })

        os.makedirs(args.output_dir, exist_ok=True)
        if args.refusal_feature_type == "constant":
            output_file = os.path.join(args.output_dir, f"{args.target_model_name}-top_{args.retain_topk}-delta_{delta}-safety.json")
        elif args.refusal_feature_type == "variable":
            output_file = os.path.join(args.output_dir, f"{args.target_model_name}-top_{args.retain_topk}-delta_{delta}-beta_{args.beta}-safety.json")
        elif args.refusal_feature_type == "mask_and_map":
            output_file = os.path.join(args.output_dir, f"{args.target_model_name}-top_{args.retain_topk}-safety.json")
        else:
            raise ValueError("Invalid refusal feature type.")
        with open(output_file, "w") as f:
            json.dump(completions, f, indent=4)

        print("Running pipeline for test dataset finished.")