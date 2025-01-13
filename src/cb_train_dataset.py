from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
import transformers
from typing import Dict
import torch
import numpy as np
from tqdm import tqdm
import json
import random
import csv
from refusal_feature.mask_run_pipeline import run_pipeline

random.seed(0)


class CircuitBreakerDataset(Dataset):

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 num_examples,
                 model_base,
                 model_name_or_path,
                 selected_layer_ids=None,
                 args=None
                 ):
        super(CircuitBreakerDataset, self).__init__()

        self.model_name_or_path = model_name_or_path.lower()
        self.max_length = 1024
        self.model = model_base
        self.selected_layer_ids = selected_layer_ids
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_on_inputs = False
        self.ignore_index = -100
        self.args = args

        if 'llama-3' in self.model_name_or_path:
            one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}<|eot_id|>"
        elif 'mistral' in self.model_name_or_path:
            one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}</s>"
        elif 'gemma' in self.model_name_or_path:
            one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
        elif 'qwen2' in self.model_name_or_path:
            one_shot_template = "{user_tag}{instruction}{assistant_tag}<SEPARATOR>{response}"
        else:
            raise NotImplementedError(f"Config {self.model_name_or_path} not found")

        # ================ Model and Template Config  ================
        # Default configs
        sep_token = ""
        switch_select = [0]
        use_refusal_retain = False
        user_tag, assistant_tag = None, None
        if 'llama-3' in self.model_name_or_path:
            print("USING LLAMA TEMPLATE")
            user_tag = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            assistant_tag = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            use_refusal_retain = True
        elif 'mistral' in self.model_name_or_path:
            print("USING MISTRAL TEMPLATE")
            # fix spacing issue in template
            user_tag = "<s> [INST]"
            assistant_tag = "[/INST]"
            sep_token = " "
            use_refusal_retain = True
        elif 'gemma' in self.model_name_or_path:
            print("USING GEMMA TEMPLATE")
            user_tag = "<bos><start_of_turn>user\n"
            assistant_tag = "<|end_of_turn|>\n<start_of_turn>model\n"
            use_refusal_retain = True
        elif 'qwen2' in self.model_name_or_path:
            print("USING QWEN2 TEMPLATE")
            user_tag = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            assistant_tag = "<|im_end|>\n<|im_start|>assistant\n"
            sep_token = " "
            use_refusal_retain = True
        else:
            raise NotImplementedError(f"Config {self.model_name_or_path} not found")

        assert user_tag and assistant_tag, "user_tag/assistant_tag not defined"

        self.user_tag = user_tag
        self.assistant_tag = assistant_tag
        self.sep_token = sep_token

        # ======================= Retain ======================= #
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", cache_dir="data/circuit-breakers")
        orig_s = []
        for example in ds:
            messages = example["messages"]
            if len(messages) < 2:
                continue

            input_output_format = one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=messages[0]["content"], response=messages[1]["content"])
            input, output = input_output_format.split("<SEPARATOR>")
            formatted_input = {"input": input, "output": output}
            orig_s.append(formatted_input)

            if len(orig_s) == num_examples:
                break
        self.orig_s_retain = orig_s
        random.shuffle(self.orig_s_retain)
        print("orig_s_retain[0]:", orig_s[0])
        print("orig_s nums:", len(self.orig_s_retain))

        # ======================= Borderline Retain ======================= #
        with open("./data/over-refusal_test/or-bench_train-adv.json") as file:
            over_refusal_dataset = json.load(file)
        orig_over_refusal = []
        for example in over_refusal_dataset:
            input_output_format = one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=example["instruction"] if 'instruction' in example else example["prompt"],
                response=example["output"])
            input, output = input_output_format.split("<SEPARATOR>")
            formatted_input = {"input": input, "output": output}
            orig_over_refusal.append(formatted_input)

        self.orig_over_refusal = orig_over_refusal
        random.shuffle(self.orig_over_refusal)
        print("orig_over_refusal[0]:", orig_over_refusal[0])
        print("orig_over_refusal nums:", len(self.orig_over_refusal))

        # ======================= Refusal Retain ======================= #
        if use_refusal_retain:
            with open("data/circuit-breakers/circuit_breakers_train.json") as file:
                dataset = json.load(file)

            random.shuffle(dataset)
            refusal_retain_orig = []
            for i, d in tqdm(enumerate(dataset)):
                formatted_input = one_shot_template.format(
                    user_tag=user_tag, assistant_tag=assistant_tag,
                    instruction=d['prompt'], response=d['llama3_output'])
                input, output = formatted_input.split("<SEPARATOR>")
                refusal_retain_orig.append({"input": input, "output": output})
                if len(refusal_retain_orig) == num_examples:
                    break

            self.orig_s_refusal = refusal_retain_orig
            random.shuffle(self.orig_s_refusal)
            print("refusal_orig_s[0]:", refusal_retain_orig[0])
            print("refusal_orig_s nums:", len(self.orig_s_refusal))

        if len(self.orig_s_retain) > len(self.orig_s_refusal):
            num = len(self.orig_s_retain) - len(self.orig_s_refusal)
            self.orig_s_refusal += random.sample(self.orig_s_refusal, num)

        # ======================= RFA Features Construction======================= #
        with open("data/build_vector/harmful_train.json") as file:
            harmful_dataset = json.load(file)
            harmful_nums = len(harmful_dataset)
        with open("data/build_vector/harmless_train.json") as file:
            harmless_dataset = json.load(file)
            harmless_nums = len(harmless_dataset)

        harmful_train, harmless_train = [], []

        for i, d in tqdm(enumerate(harmful_dataset)):
            formatted_input = one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=d['instruction'], response="").split("<SEPARATOR>")[0]

            harmful_train.append(formatted_input)

        for i, d in tqdm(enumerate(harmless_dataset)):
            formatted_input = one_shot_template.format(
                user_tag=user_tag, assistant_tag=assistant_tag,
                instruction=d['instruction'], response="").split("<SEPARATOR>")[0]

            harmless_train.append(formatted_input)

        print("harmful_dataset[0]:", harmful_train[0])
        print("harmless_dataset[0]:", harmless_train[0])
        print("harmful_nums/harmless_nums:", harmful_nums, harmless_nums)

        feature_tokenized_kwargs = dict(max_length=64, padding='max_length', truncation=True, return_tensors="pt", add_special_tokens=False)
        self.harmful_data_tokenized = [self.tokenizer(one_harmful_instance, **feature_tokenized_kwargs) for one_harmful_instance in harmful_train]
        self.harmless_data_tokenized = [self.tokenizer(one_harmless_instance, **feature_tokenized_kwargs) for one_harmless_instance in harmless_train]

    def __len__(self):
        return len(self.orig_s_retain)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        text_input, text_output = self.orig_s_retain[i]["input"], self.orig_s_retain[i]["output"]
        refusal_input, refusal_output = self.orig_s_refusal[i]["input"], self.orig_s_refusal[i]["output"]
        over_refusal_nums = len(self.orig_over_refusal)
        selected_index = i % over_refusal_nums
        over_refusal_input, over_refusal_output = self.orig_over_refusal[selected_index]["input"], self.orig_over_refusal[selected_index]["output"]
        max_length = 1024

        text_input_ids = self.tokenizer.encode(text_input, add_special_tokens=False)
        text_output_ids = self.tokenizer.encode(text_output, add_special_tokens=False)
        input_ids = text_input_ids + text_output_ids
        labels = [self.ignore_index] * len(text_input_ids) + text_output_ids
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        attention_mask = [1] * len(input_ids)

        refusal_text_input_ids = self.tokenizer.encode(refusal_input, add_special_tokens=False)
        refusal_text_output_ids = self.tokenizer.encode(refusal_output, add_special_tokens=False)
        refusal_input_ids = refusal_text_input_ids + refusal_text_output_ids
        refusal_labels = [self.ignore_index] * len(refusal_text_input_ids) + refusal_text_output_ids
        refusal_input_ids = refusal_input_ids[:max_length]
        refusal_labels = refusal_labels[:max_length]
        refusal_attention_mask = [1] * len(refusal_input_ids)

        over_refusal_text_input_ids = self.tokenizer.encode(over_refusal_input, add_special_tokens=False)
        over_refusal_text_output_ids = self.tokenizer.encode(over_refusal_output, add_special_tokens=False)
        over_refusal_input_ids = over_refusal_text_input_ids + over_refusal_text_output_ids
        over_refusal_labels = [self.ignore_index] * len(over_refusal_text_input_ids) + over_refusal_text_output_ids
        over_refusal_input_ids = over_refusal_input_ids[:max_length]
        over_refusal_labels = over_refusal_labels[:max_length]
        over_refusal_attention_mask = [1] * len(over_refusal_input_ids)

        data = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            refusal_input_ids=refusal_input_ids,
            refusal_attention_mask=refusal_attention_mask,
            refusal_labels=refusal_labels,
            over_refusal_input_ids=over_refusal_input_ids,
            over_refusal_attention_mask=over_refusal_attention_mask,
            over_refusal_labels=over_refusal_labels
        )

        return data

    def get_refusal_feature(self, step):

        return run_pipeline(step=step,
                            harmful_data=self.harmful_data_tokenized,
                            harmless_data=self.harmless_data_tokenized,
                            model_base=self.model.base_model.model,
                            selected_layer_ids=self.selected_layer_ids,
                            args=self.args
                            )
