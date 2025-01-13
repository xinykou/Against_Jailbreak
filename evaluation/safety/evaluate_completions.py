import argparse
import json
import os

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ["OPENAI_API_KEY"] = "sk-"
from tqdm import tqdm
import numpy as np

from vllm import LLM, SamplingParams

from eval_utils import compute_results_classifier

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--model_name", type=str, default="", help="The name of the model to evaluate")
    parser.add_argument("--cls_path", type=str, default='cais/HarmBench-Llama-2-13b-cls',
                        help="The name or path of the classifier for computing ASR")
    parser.add_argument("--behaviors_path", type=str, default='',
                        help="The path to the behaviors file")
    parser.add_argument("--completions_path", type=str,
                        help="The path to the completions to evaluate")
    parser.add_argument("--save_path", type=str,
                        help="The path for saving results")
    parser.add_argument("--include_advbench_metric", action="store_true",
                        help="Whether to include the AdvBench refusal metric")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="The number of tokens to evaluate")
    parser.add_argument("--generate_with_vllm", action="store_true", )
    parser.add_argument("--record_mask", action="store_true",
                        help="Whether to record the mask when ablations are used")
    parser.add_argument("--top_k", type=int, default=None,
                        help="The number of top-k completions to evaluate")
    parser.add_argument("--delta", type=float, default=None, help="The lambda value for the safety metric")
    parser.add_argument("--safety_evaluator_name",
                        type=str,
                        default="HarmBench-Llama-2-13b-cls",
                        choices=["HarmBench-Llama-2-13b-cls", "Llama-Guard-3-8B",
                                "ASR-Keywords-1", "ASR-Keywords-2",
                                "glm-4-plus", "glm-4-flash"
                                ]
                        )
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    print(args)

    # ========== initialize results ========== #
    with open(args.completions_path, 'r') as f:
        completions_list = json.load(f)

    tokenizer = None
    # Clip the 'generation' field of the completions to have a maximum of max_new_tokens tokens
    # for completion in completions_list:
    #     generation = completion['output']
    #     tokenized_text = tokenizer.encode(generation, max_length=args.max_new_tokens, truncation=True)
    #     clipped_generation = tokenizer.decode(tokenized_text, skip_special_tokens=True)
    #     completion['output'] = clipped_generation

    cls = None
    cls_params = None
    # ========== evaluate completions ========== #
    if args.safety_evaluator_name == "Llama-Guard-3-8B" or args.safety_evaluator_name == "HarmBench-Llama-2-13b-cls":
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.cls_path)
        tokenizer.truncation_side = "right"

        if args.generate_with_vllm:
            cls = LLM(model=args.cls_path,
                      tensor_parallel_size=2,
                      dtype="bfloat16",
                      gpu_memory_utilization=0.9
                      )

            # cls.llm_engine.tokenizer.tokenizer.truncation_side = "left"

            cls_params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
        else:
            cls = AutoModelForCausalLM.from_pretrained(args.cls_path,
                                                       device_map="auto",
                                                       torch_dtype=torch.bfloat16)
            cls_params = GenerationConfig(max_new_tokens=args.max_new_tokens, num_beams=1, temperature=0.0, do_sample=False)

    results = {}
    error_num = 0
    for item_data in tqdm(completions_list, total=len(completions_list), desc="Evaluating completions"):
        current_results = compute_results_classifier(item_data, cls, cls_params,
                                                     tokenizer=tokenizer,
                                                     evaluator_name=args.safety_evaluator_name)
        # todo: if args.include_advbench_metric:
        # current_results = compute_results_advbench(current_results, item_data)
        if current_results == -1:
            error_num += 1
            continue

        item_data['attacked_or_not'] = current_results
        if 'category' not in item_data:
            item_data['category'] = 'unknown'
        if item_data['category'] not in results:
            results[item_data['category']] = []

        results[item_data['category']].append(current_results)

    sub_success_num_records = []
    sub_success_rate_records = []
    for i, (behavior_id, data) in enumerate(results.items()):
        sub_success_nums = np.sum(data)
        sub_success_rate = round(sub_success_nums / len(data), 3)
        sub_asr = f"{i}. {behavior_id} === average asr {sub_success_rate}"
        print(sub_asr)
        sub_success_num_records.append(sub_success_nums)
        sub_success_rate_records.append(sub_asr)

    print(f"Error number: {error_num}")

    total_success_rate = round(sum(sub_success_num_records) / len(completions_list), 3)
    aver_asr = f"Average ASR: {total_success_rate}"
    print(aver_asr)

    completions_list.extend(sub_success_rate_records)
    completions_list.append(aver_asr)
    # Make dirs to output_file if not yet exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True) if os.path.dirname(args.save_path) else None
    with open(args.save_path, 'w') as file:
        json.dump(completions_list, file, indent=4)

    print("Results saved to", args.save_path)

    # ==========ablations:  record mask when ablations are used ========== #
    main_path = os.path.dirname(args.save_path)
    safety_record_path = os.path.join(main_path, args.model_name + '_safety_record.json')
    if args.record_mask:
        if args.top_k is None or args.delta is None:
            raise ValueError("top_k and delta must be set when record_mask is True")

        if os.path.exists(safety_record_path):
            with open(safety_record_path, 'r') as f:
                safety_records_dict = json.load(f)
                safety_records_dict[f"top_{args.top_k}-delta_{args.delta}"] = total_success_rate
        else:
            safety_records_dict = {f"top_{args.top_k}-delta_{args.delta}": total_success_rate}

        with open(safety_record_path, 'w') as f:
            json.dump(safety_records_dict, f, indent=4)


if __name__ == "__main__":
    main()
