import os
import json
import torch
from transformers import LlavaNextForConditionalGeneration, AutoModelForCausalLM

def save_model_and_tokenizer(model_name_or_path, model, tokenizer, output_dir, trainer):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n\nModel and tokenizer saving to {output_dir}\n\n")
    
    # merge lora
    merged_model = model.merge_and_unload()

    merged_path = os.path.join(output_dir, "merged")
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    lorra_config_path = os.path.join(merged_path, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)
    

def save_llava_model_and_tokenizer(model_name_or_path, model, processor, drop_layers_after, output_dir, trainer):
    os.makedirs(output_dir, exist_ok=True)
    print(f"MModel and processor saving to {output_dir}")
    
    # merge lora
    merged_model = model.merge_and_unload() 
    # merge original layers
    
    anchor_model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=merged_model.dtype)
    merged_model.language_model.model.layers = merged_model.language_model.model.layers + anchor_model.language_model.model.layers[drop_layers_after+1:]
    merged_model.config = anchor_model.config

    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    lorra_config_path = os.path.join(output_dir, "lorra_config.json")
    with open(lorra_config_path, "w", encoding="utf-8") as file:
        json.dump(trainer.lorra_args.to_dict(), file, indent=2)
    
    torch.use_deterministic_algorithms(False)
    if trainer.training_args.do_eval:
        trainer.evaluate()
    