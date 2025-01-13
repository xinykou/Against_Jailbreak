import transformers
import json
import argparse
import os
os.environ["OPENAI_API_KEY"] = "sk-"
import csv
from tqdm import tqdm
import torch
from baseline import get_template, load_model_and_tokenizer, load_vllm_model
from api_models import api_models_map
from functools import partial
from vllm import SamplingParams, LLM
from accelerate.utils import find_executable_batch_size

import yaml
import random
from safetensors import safe_open
import numpy as np
# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--dataset_name", type=str, default="HarmBench")
    parser.add_argument("--model_name", type=str,
                        help="The name of the model in the models config file")
    parser.add_argument("--models_config_file", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--attack_type", type=str, default='no_attack',
                        help="The path to the behaviors file")
    parser.add_argument("--test_cases_path", type=str,
                        help="The path to the test cases file to generate completions for")
    parser.add_argument("--reference_path", type=str,
                        default="",
                        help="The path to the reference categories file")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving completions")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max new tokens for completions")
    parser.add_argument("--generate_with_vllm", action="store_true",
                        help="Whether to generate completions with vLLM (if applicable)")
    parser.add_argument("--vllm_block_size", type=int, default=10,
                        help="Block size for vLLM generation")
    parser.add_argument("--selected_layer_ids", type=str, default="",
                        help="Selected layer ids for vLLM generation")
    parser.add_argument("--save_mask_path", type=str, default=None,
                        help="The path for saving layer mask positions")
    parser.add_argument("--ablation_coefficient", type=float, default=1.0,
                        help="The coefficient for ablation for scav, default is 1.0")

    parser.add_argument("--record_time", action="store_true",
                        help="Whether to record time for each generation"
                        )
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    # Load model config file
    config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
    with open(config_file) as file:
        all_configs = yaml.full_load(file)

    num_gpus = all_configs[args.model_name].get('num_gpus', 0)
    # check num gpus available to see if greater than num_gpus in config
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available != num_gpus:
        print(
            f"Warning: num_gpus in config ({num_gpus}) does not match num_gpus available ({num_gpus_available}). Using {num_gpus_available} GPUs.")
        num_gpus = num_gpus_available
    model_config = all_configs[args.model_name]['model']
    model_config['num_gpus'] = num_gpus

    print("model_config", model_config)

    if args.dataset_name == "HarmBench" or args.dataset_name == "AdvBench" or args.dataset_name == "JailbreakBench":
        pass
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

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

        if args.model_name == "llama3_8B" or args.model_name == "mistral_7B":
            layer_nums = 32
            mask_layer_ids = []
            for i in range(layer_nums):
                if str(i) not in selected_layer_ids:
                    mask_layer_ids.append(i)
        elif args.model_name == "gemma_7B":
            layer_nums = 28
            mask_layer_ids = []
            for i in range(layer_nums):
                if str(i) not in selected_layer_ids:
                    mask_layer_ids.append(i)
        else:
            raise ValueError(f"Model {args.model_name} not supported")

    # Load test cases
    if args.attack_type == 'no_attack':
        if args.test_cases_path.endswith('.csv'):  # jailbreakbench
            with open(args.test_cases_path, 'r') as f:
                reader = csv.reader(f)
                test_cases_ = list(reader)

            test_cases_data = []
            for data in test_cases_:
                test_cases_data.append({"instruction": data[1], "category": "default"})

        elif args.test_cases_path.endswith('.json'):
            with open(args.test_cases_path, 'r') as f:
                test_cases_data = json.load(f)
    elif args.attack_type == 'GCG':
        test_cases_data = []
        if args.reference_path != "":
            with open(args.reference_path, 'r') as f:
                ref_data = json.load(f)
        with open(args.test_cases_path, 'r') as f:
            org_test_cases_data = json.load(f)
            for value in org_test_cases_data.values():
                attacked_prompt = value[0]
                if args.reference_path == "":
                    instance = {
                        "instruction": attacked_prompt,
                        "category": "default"
                    }
                else:
                    for ref_item in ref_data:
                        if ref_item["instruction"] in attacked_prompt:
                            instance = {
                                "instruction": attacked_prompt,
                                "category": ref_item["category"]
                            }
                            break
                try:
                    test_cases_data.append(instance)
                except:
                    raise ValueError(f"Reference category not found for {attacked_prompt}")

    elif args.attack_type == 'PAIR' or args.attack_type == "GPTFuzz":
        test_cases_data = []
        with open(args.test_cases_path, 'r') as f:
            org_test_cases_data = json.load(f)
            for value in org_test_cases_data.values():
                instance = {
                    "instruction": value[0] if args.attack_type == "PAIR" else value[0][0],
                    "category": "default"
                }
                test_cases_data.append(instance)

    elif args.attack_type == 'Adaptive_Attack' or args.attack_type == 'ChatBug':
        test_cases_data = []
        if args.reference_path != "":
            with open(args.reference_path, 'r') as f:
                ref_data = json.load(f)
        with open(args.test_cases_path, 'r') as f:
            org_test_cases_data = json.load(f)
            for item in org_test_cases_data:
                attacked_prompt = item["jailbreak_prompt"]
                if args.reference_path != "":
                    for ref_item in ref_data:
                        if ref_item["instruction"].lower() in attacked_prompt.lower():
                            instance = {
                                "instruction": attacked_prompt,
                                "category": ref_item["category"]
                            }
                            break
                else:
                    instance = {
                        "instruction": attacked_prompt,
                        "category": "default"
                    }

                try:
                    test_cases_data.append(instance)
                except:
                    raise ValueError(f"Reference category not found for {attacked_prompt}")

    elif args.attack_type == 'HumanJailbreaks':
        test_cases_data = []
        with open(args.test_cases_path, 'r') as f:
            org_test_cases_data = json.load(f)
            for value_list in org_test_cases_data.values():
                for value in value_list:
                    instance = {
                        "instruction": value,
                        "category": "default"
                    }
                test_cases_data.append(instance)
        # random.seed(0)
        # random.shuffle(test_cases_data)
        # test_cases_data = test_cases_data[-100:]
    else:
        raise ValueError(f"Attack type {args.attack_type} not supported")
    print(f"Total test cases: {len(test_cases_data)}")


    use_template = True
    if args.attack_type == 'ChatBug':
        use_template = False
    generation_function = load_generation_function(model_config,
                                                   args.max_new_tokens,
                                                   use_template=use_template,
                                                   test_cases_path=args.test_cases_path,
                                                   generate_with_vllm=args.generate_with_vllm,
                                                   model_name=args.model_name,
                                                   classifier_path=all_configs[args.model_name].get('classifier_path', None),
                                                   mask_layer_ids=mask_layer_ids,
                                                   use_masked_scav=use_masked_scav,
                                                   map_current_direction_mask_dict=map_current_direction_mask_dict,
                                                   use_ablation_coefficient=use_ablation_coefficient,
                                                   ablation_coefficient=args.ablation_coefficient
                                                   )

    test_cases_ = []
    for index in range(len(test_cases_data)):
        instr = test_cases_data[index]["instruction"]
        test_cases_.append(instr)

    if len(test_cases_) == 0:
        print('No test cases to generate completions for')
        return
    else:
        print(f'Generating completions for {len(test_cases_)} test cases')

    # ==== Generate ====
    print('Generating completions...')
    if args.vllm_block_size < len(test_cases_):
        generations = []
        inference_times = []
        for i in range(0, len(test_cases_), args.vllm_block_size):
            block_test_cases = test_cases_[i:i + args.vllm_block_size] if len(
                test_cases_[i:]) >= args.vllm_block_size else test_cases_[i:]
            if args.record_time:
                block_generations, batch_inference_time = generation_function(test_cases=[t for t in block_test_cases], record_time=True)
                inference_times.append(batch_inference_time)
            else:
                block_generations = generation_function(test_cases=[t for t in block_test_cases])
            generations.extend(block_generations)
        if args.record_time:
            mean_inference_time = np.mean(inference_times)
            mean_inference_time = round(mean_inference_time, 3)
            variance_inference_time = np.var(inference_times)
            variance_inference_time = round(variance_inference_time, 3)
            print(f'Mean inference time: {mean_inference_time} seconds')
            print(f'Variance of inference time: {variance_inference_time} seconds')
    else:
        generations = generation_function(test_cases=test_cases_)

    returned_data = []
    for index in range(len(test_cases_data)):
        instr = test_cases_data[index]["instruction"]
        category = test_cases_data[index]["category"] if args.attack_type != 'PAIR' else "default"
        output = generations[index]
        returned_data.append({"instruction": instr, "category": category, "output": output})
    print('Done')

    # Create directories for save_path
    print(f'Saving completions to {args.save_path}...')
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    with open(args.save_path, 'w') as file:
        json.dump(returned_data, file, indent=4)
    print('Saved')


    if args.record_time:
        main_path = args.save_path.split('inference_time')[0] + 'inference_time'
        file_name = os.path.join(main_path, "inference_time_record.json")
        if not os.path.exists(main_path):
            os.makedirs(main_path)
            record_dict = {}
            sub_record_dict = {"mean_inference_time": mean_inference_time,
                               "variance_inference_time": variance_inference_time}
            record_dict[file_name] = sub_record_dict
        else:
            with open(file_name, 'r') as file:
                record_dict = json.load(file)
                sub_record_dict = {"mean_inference_time": mean_inference_time,
                                   "variance_inference_time": variance_inference_time}
                record_dict[args.save_path] = sub_record_dict

        with open(file_name, 'w') as file:
            json.dump(record_dict, file, indent=4)

def _vllm_generate(model, test_cases, template, use_template, **generation_kwargs):
    if use_template:
        inputs = [template['prompt'].format(instruction=s) for s in test_cases]
    else:
        inputs = test_cases
    outputs = model.generate(inputs, **generation_kwargs)
    generations = [o.outputs[0].text.strip() for o in outputs]
    return generations



def _api_model_generate(model, test_cases, **generation_kwargs):
    # MultiModal test cases are in saved in pair of [[img, text], ...]
    if isinstance(test_cases[0], (tuple, list)):
        images, prompts = zip(*test_cases)
        return model.generate(prompts=list(prompts), images=list(images), **generation_kwargs)
    return model.generate(prompts=test_cases, **generation_kwargs)


def load_generation_function(model_config, max_new_tokens,
                             test_cases_path=None,
                             generate_with_vllm=False,
                             use_template=True,
                             model_name=None,
                             classifier_path=None,
                             mask_layer_ids=None,
                             use_masked_scav=False,
                             map_current_direction_mask_dict={},
                             use_ablation_coefficient=False,
                             ablation_coefficient=1.0):
    model_name_or_path = model_config['model_name_or_path']

    if (model := api_models_map(**model_config)):
        generation_kwargs = dict(max_new_tokens=max_new_tokens, temperature=0.0, use_tqdm=True)
        return partial(_api_model_generate, model=model, **generation_kwargs)

    print('--------------------------------Using HF generation----------------------')
    print(f"model_name_or_path: {classifier_path}")
    from scav.perturbation import Perturbation
    from scav.model_generation import ModelGeneration
    clfr = torch.load(classifier_path)
    pert = Perturbation(clfr, target_probability=0.05,
                        layer_mask=mask_layer_ids,
                        use_masked_scav=use_masked_scav,
                        map_current_direction_mask_dict=map_current_direction_mask_dict,
                        use_ablation_coefficient=use_ablation_coefficient,
                        ablation_coefficient=ablation_coefficient
                        )
    llm_gen = ModelGeneration(model_name)
    llm_gen.set_perturbation(pert)
    return partial(llm_gen.batch_generate,
                   model=llm_gen.model,
                   max_length=max_new_tokens,
                   output_hidden_states=False,
                   use_template=use_template)


if __name__ == "__main__":
    main()
