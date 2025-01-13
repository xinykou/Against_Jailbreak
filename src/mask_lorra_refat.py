import sys
from functools import partial
import logging
import os
import json
import gc
import atexit
import numpy as np
# os.environ["HTTP_PROXY"] = "127.0.0.1:27888"
# os.environ["HTTPS_PROXY"] = "127.0.0.1:27888"
# from deepspeed import zero
# from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from peft import LoraConfig, get_peft_model
import transformers
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM, deepspeed
from custom_trainer import CustomTrainer
import torch
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from DataCollator import DataCollatorForSeq2Seq

from cb_train_dataset import (
    CircuitBreakerDataset
)

from utils import save_model_and_tokenizer
from args import (
    ModelArguments,
    TrainingArguments, 
    LoraArguments, 
    LorraArguments,
)
import argparse


def train(args=None):
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
    )
    config_file = args.our_config_file
    (
        model_args,
        training_args,
        lora_args,
        lorra_args,
    ) = parser.parse_yaml_file(config_file)

    print(lorra_args.to_dict())
    print(lora_args)
    print(model_args)
    print(training_args)

    device_map = "auto"
    if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
        logging.warning(
            "FSDP and ZeRO3 are both currently incompatible with QLoRA."
        )

    model_name_or_path = model_args.model_name_or_path
    target_layers = lorra_args.target_layers
    transform_layers = lorra_args.transform_layers

    lorra_target_layers = [int(layer) for layer in target_layers.split(",")] # target representations
    if "start_to_end" in transform_layers:
        lora_layers_to_transform = [i for i in range(lorra_target_layers[0], lorra_target_layers[1])]
    elif "discrete" in transform_layers:
        lora_layers_to_transform = [int(layer) for layer in lorra_target_layers]
    else:
        raise NotImplementedError("Only start_to_end and discrete are supported for now")

    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=lora_layers_to_transform,
        task_type="CAUSAL_LM",
    )

    print("lorra_transform_layers", lora_layers_to_transform)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    extra_save_kargs = dict(tokenizer=tokenizer)
    save_model_function = save_model_and_tokenizer

    # if training_args.deepspeed is not None:
    model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            cache_dir=training_args.cache_dir,
            device_map=device_map,
    )
    # else:
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name_or_path,
    #         torch_dtype=torch.bfloat16,
    #         cache_dir=training_args.cache_dir
    #     )
    save_model_function = partial(save_model_function, 
                    model_name_or_path=model_name_or_path,
                    output_dir=training_args.output_dir,
                    **extra_save_kargs)

    print(lora_args.lora_target_modules, lora_layers_to_transform)

    model = get_peft_model(model, lora_config)
    print("model", model)


    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_dataset = CircuitBreakerDataset(tokenizer,
                                          num_examples=5000,
                                          model_base=model,
                                          model_name_or_path=model_name_or_path,
                                          selected_layer_ids=lora_layers_to_transform,
                                          args=training_args)
    print("TRAIN LEN: ", len(train_dataset))

    training_args.remove_unused_columns = False

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    trainer = CustomTrainer(
        lorra_args=lorra_args, model=model, tokenizer=tokenizer,
        args=training_args, train_dataset=train_dataset, data_collator=data_collator,
        selected_layer_ids=lora_layers_to_transform
    )
    model.config.use_cache = False
    atexit.register(save_model_function, model=model, trainer=trainer)
    trainer.train()



if __name__ == "__main__":
    SEED = 42
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--our_config_file", type=str, required=True)
    args = parser.parse_args()
    train(args)