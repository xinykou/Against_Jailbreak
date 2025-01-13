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
import time

import yaml
import random
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
    parser.add_argument("--record_time", action="store_true",
                        help="Whether to record inference time")
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    # Load model config file
    config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
    with open(config_file) as file:
        model_configs = yaml.full_load(file)

    num_gpus = model_configs[args.model_name].get('num_gpus', 0)
    # check num gpus available to see if greater than num_gpus in config
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available != num_gpus:
        print(
            f"Warning: num_gpus in config ({num_gpus}) does not match num_gpus available ({num_gpus_available}). Using {num_gpus_available} GPUs.")
        num_gpus = num_gpus_available
    model_config = model_configs[args.model_name]['model']
    model_config['num_gpus'] = num_gpus

    print("model_config", model_config)

    if args.dataset_name == "HarmBench":
        pass
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

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

    elif args.attack_type == 'GCG' or args.attack_type == 'AutoDAN':
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
                    "instruction": value[0] if args.attack_type == 'PAIR' else value[0][0],
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
                                                   generate_with_vllm=args.generate_with_vllm
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
                block_generations, inference_time = generation_function(test_cases=[t for t in block_test_cases],
                                                                         record_time=True)
                inference_times.append(inference_time)
            else:
                block_generations = generation_function(test_cases=[t for t in block_test_cases])
            generations.extend(block_generations)

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
        print(f'Saving inference time to {file_name}...')
        if not os.path.exists(file_name):
            dir_name = os.path.dirname(file_name)
            os.makedirs(dir_name, exist_ok=True)
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


def _hf_generate_with_batching(model, tokenizer, test_cases, template, record_time=False, **generation_kwargs):
    @find_executable_batch_size(starting_batch_size=len(test_cases))
    def inner_generation_loop(batch_size):
        nonlocal model, tokenizer, test_cases, template, generation_kwargs
        generations = []
        start_time = time.time()
        for i in tqdm(range(0, len(test_cases), batch_size)):

            batched_test_cases = test_cases[i:i + batch_size]
            inputs = [template['prompt'].format(instruction=s) for s in batched_test_cases]
            inputs = tokenizer(inputs, return_tensors='pt', padding=True)
            inputs = inputs.to(model.device)
            with torch.no_grad():
                outputs = model.generate(inputs=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                         **generation_kwargs).cpu()

            generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            batch_generations = [tokenizer.decode(o, skip_special_tokens=True).strip() for o in generated_tokens]
            generations.extend(batch_generations)

        end_time = time.time()
        inference_time = end_time - start_time
        if record_time:
            return generations, inference_time
        else:
            return generations

    return inner_generation_loop()


def _api_model_generate(model, test_cases, **generation_kwargs):
    # MultiModal test cases are in saved in pair of [[img, text], ...]
    if isinstance(test_cases[0], (tuple, list)):
        images, prompts = zip(*test_cases)
        return model.generate(prompts=list(prompts), images=list(images), **generation_kwargs)
    return model.generate(prompts=test_cases, **generation_kwargs)


def load_generation_function(model_config, max_new_tokens, test_cases_path=None, generate_with_vllm=False, use_template=True):
    model_name_or_path = model_config['model_name_or_path']

    if (model := api_models_map(**model_config)):
        generation_kwargs = dict(max_new_tokens=max_new_tokens, temperature=0.0, use_tqdm=True)
        return partial(_api_model_generate, model=model, **generation_kwargs)
    elif generate_with_vllm:
        print('Using vLLM generation')
        sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        model = LLM(model=model_name_or_path,
                    tensor_parallel_size=model_config.get('num_gpus', 1),
                    dtype=model_config.get('dtype', torch.float32),
                    gpu_memory_utilization=0.95
                    )

        generation_kwargs = dict(sampling_params=sp, use_tqdm=True)
        TEMPLATE = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))
        return partial(_vllm_generate, model=model, template=TEMPLATE, use_template=use_template,
                       **generation_kwargs)
    else:
        print('Using HF generation')
        model, tokenizer = load_model_and_tokenizer(**model_config)
        tokenizer.pad_token = tokenizer.eos_token
        generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
        TEMPLATE = get_template(model_name_or_path, chat_template=model_config.get('chat_template', None))
        return partial(_hf_generate_with_batching, model=model, tokenizer=tokenizer, template=TEMPLATE,
                       **generation_kwargs)


if __name__ == "__main__":
    main()
