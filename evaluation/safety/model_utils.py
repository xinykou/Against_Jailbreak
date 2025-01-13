import os
import torch
import random
import json
from typing import Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from vllm import LLM
from huggingface_hub import login as hf_login
import ray
from fastchat.model import get_conversation_template
from fastchat.conversation import get_conv_template
from inspect import signature

LLAMA3_CHAT_PROMPT = {
    "description": "Template used by Llama3 Chat",
    "prompt": "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"+"<|start_header_id|>assistant<|end_header_id|>\n\n"
}

MISTRAL_PROMPT = {
    "description": "Template used by Mistral Instruct",
    "prompt": "<s> [INST] {instruction} [/INST]"
}

GEMMA_PROMPT = {
    "description": "Template used by Gemma Instruct",
    "prompt": "<bos>[start_of_turn]user\n{instruction}[end_of_turn]\n<start_of_turn>model\n"
}

QWEN_CHAT_PROMPT = {
    "description": "Template used by Qwen Chat",
    "prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
}
########## CHAT TEMPLATE ###########

def get_template(model_name_or_path=None, chat_template=None, fschat_template=None, system_message=None, return_fschat_conv=False, **kwargs):
    # ==== First check for fschat template ====
    if fschat_template or return_fschat_conv:
        fschat_conv = _get_fschat_conv(model_name_or_path, fschat_template, system_message)
        if return_fschat_conv: 
            print("Found FastChat conv template for", model_name_or_path)
            print(fschat_conv.dict())
            return fschat_conv
        else:
            fschat_conv.append_message(fschat_conv.roles[0], "{instruction}")
            fschat_conv.append_message(fschat_conv.roles[1], None) 
            TEMPLATE = {"description": f"fschat template {fschat_conv.name}", "prompt": fschat_conv.get_prompt()}
    # ===== Check for some older chat model templates ====
    elif chat_template == "llama-3":
        TEMPLATE = LLAMA3_CHAT_PROMPT
    elif chat_template == "mistral" or chat_template == "mixtral":
        TEMPLATE = MISTRAL_PROMPT
    elif chat_template == "gemma":
        TEMPLATE = GEMMA_PROMPT
    elif chat_template == "qwen2":
        TEMPLATE = QWEN_CHAT_PROMPT
    elif chat_template == "wizard":
        TEMPLATE = VICUNA_PROMPT
    elif chat_template == "vicuna":
        TEMPLATE = VICUNA_PROMPT
    elif chat_template == "oasst":
        TEMPLATE = OASST_PROMPT
    elif chat_template == "oasst_v1_1":
        TEMPLATE = OASST_PROMPT_v1_1
    elif chat_template == "llama-2":
        TEMPLATE = LLAMA2_CHAT_PROMPT
    elif chat_template == "falcon_instruct": #falcon 7b / 40b instruct
        TEMPLATE = FALCON_INSTRUCT_PROMPT
    elif chat_template == "falcon_chat": #falcon 180B_chat
        TEMPLATE = FALCON_CHAT_PROMPT
    elif chat_template == "mpt":
        TEMPLATE = MPT_PROMPT
    elif chat_template == "koala":
        TEMPLATE = KOALA_PROMPT
    elif chat_template == "dolly":
        TEMPLATE = DOLLY_PROMPT
    elif chat_template == "internlm":
        TEMPLATE = INTERNLM_PROMPT
    elif chat_template == "orca-2":
        TEMPLATE = ORCA_2_PROMPT
    elif chat_template == "baichuan2":
        TEMPLATE = BAICHUAN_CHAT_PROMPT
    elif chat_template == "zephyr_7b_robust":
        TEMPLATE = ZEPHYR_ROBUST_PROMPT
    else:
        # ======== Else default to tokenizer.apply_chat_template =======
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            template = [{'role': 'system', 'content': system_message}, {'role': 'user', 'content': '{instruction}'}] if system_message else [{'role': 'user', 'content': '{instruction}'}]
            prompt = tokenizer.apply_chat_template(template, tokenize=False, add_generation_prompt=True)
            # Check if the prompt starts with the BOS token
            # removed <s> if it exist (LlamaTokenizer class usually have this) as our baselines will add these if needed later
            if tokenizer.bos_token and prompt.startswith(tokenizer.bos_token):
                prompt = prompt.replace(tokenizer.bos_token, "")
            TEMPLATE = {'description': f"Template used by {model_name_or_path} (tokenizer.apply_chat_template)", 'prompt': prompt}
        except:    
            assert TEMPLATE, f"Can't find instruction template for {model_name_or_path}, and apply_chat_template failed."

    print("Found Instruction template for", model_name_or_path)
    print(TEMPLATE)
        
    return TEMPLATE

def _get_fschat_conv(model_name_or_path=None, fschat_template=None, system_message=None, **kwargs):
    template_name = fschat_template
    if template_name is None:
        template_name = model_name_or_path
        print(f"WARNING: default to fschat_template={template_name} for model {model_name_or_path}")
        template = get_conversation_template(template_name)
    else:
        template = get_conv_template(template_name)
    
    # New Fschat version remove llama-2 system prompt: https://github.com/lm-sys/FastChat/blob/722ab0299fd10221fa4686267fe068a688bacd4c/fastchat/conversation.py#L1410
    if template.name == 'llama-2' and system_message is None:
        print("WARNING: using llama-2 template without safety system promp")
    
    if system_message:
        template.set_system_message(system_message)

    assert template and template.dict()['template_name'] != 'one_shot', f"Can't find fschat conversation template `{template_name}`. See https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py for supported template"
    
    return template


########## MODEL ###########

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "auto": "auto"
}


def load_model_and_tokenizer(
    model_name_or_path,
    dtype='auto',
    device_map='auto',
    trust_remote_code=False,
    revision=None,
    token=None,
    num_gpus=1,
    ## tokenizer args
    use_fast_tokenizer=True,
    padding_side='left',
    legacy=False,
    pad_token=None,
    eos_token=None,
    ## dummy args passed from get_template()
    chat_template=None,
    fschat_template=None,
    system_message=None,
    return_fschat_conv=False,
    **model_kwargs
):  
    if token:
        hf_login(token=token)
    

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
        torch_dtype=_STR_DTYPE_TO_TORCH_DTYPE[dtype], 
        device_map=device_map, 
        trust_remote_code=trust_remote_code, 
        revision=revision, 
        **model_kwargs).eval()
    
    # Init Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast_tokenizer,
        trust_remote_code=trust_remote_code,
        legacy=legacy,
        padding_side=padding_side,
    )
    if pad_token:
        tokenizer.pad_token = pad_token
    if eos_token:
        tokenizer.eos_token = eos_token

    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Tokenizer.pad_token is None, setting to tokenizer.unk_token")
        tokenizer.pad_token = tokenizer.unk_token
        print("tokenizer.pad_token", tokenizer.pad_token)
    
    return model, tokenizer


def load_vllm_model(
    model_name_or_path,
    dtype='auto',
    trust_remote_code=False,
    download_dir=None,
    revision=None,
    token=None,
    quantization=None,
    num_gpus=1,
    ## tokenizer_args
    use_fast_tokenizer=True,
    pad_token=None,
    eos_token=None,
    **kwargs
):
    if token:
        hf_login(token=token)

    if num_gpus > 1:
        _init_ray(reinit=False)
    
    # make it flexible if we want to add anything extra in yaml file
    model_kwargs = {k: kwargs[k] for k in kwargs if k in signature(LLM).parameters}
    model = LLM(model=model_name_or_path, 
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                download_dir=download_dir,
                revision=revision,
                quantization=quantization,
                tokenizer_mode="auto" if use_fast_tokenizer else "slow",
                tensor_parallel_size=num_gpus)
    
    if pad_token:
        model.llm_engine.tokenizer.tokenizer.pad_token = pad_token
    if eos_token:
        model.llm_engine.tokenizer.tokenizer.eos_token = eos_token

    return model

def _init_ray(num_cpus=8, reinit=False, resources={}):
    from transformers.dynamic_module_utils import init_hf_modules

    # check if ray already started
    if ('RAY_ADDRESS' in os.environ or ray.is_initialized()) and not reinit:
        return
    # Start RAY
    # config different ports for ray head and ray workers to avoid conflict when running multiple jobs on one machine/cluster
    # docs: https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-networking-caveats
    num_cpus = min([os.cpu_count(), num_cpus])

    os.environ['RAY_DEDUP_LOGS'] = '0'
    RAY_PORT = random.randint(0, 999) + 6000 # Random port in 6xxx zone
    RAY_MIN_PORT = random.randint(0, 489) * 100 + 10002 
    RAY_MAX_PORT = RAY_MIN_PORT + 99 # Random port ranges zone
    
    os.environ['RAY_ADDRESS'] = f"127.0.0.1:{RAY_PORT}"
    resources_args = ""
    if resources:
        # setting custom resources visbile: https://discuss.ray.io/t/access-portion-of-resource-assigned-to-task/13869
        # for example: this can be used in  setting visible device for run_pipeline.py
        os.environ['RAY_custom_unit_instance_resources'] = ",".join(resources.keys())
        resources_args = f" --resources '{json.dumps(resources)}'"
    ray_start_command = f"ray start --head --num-cpus={num_cpus} --port {RAY_PORT} --min-worker-port={RAY_MIN_PORT} --max-worker-port={RAY_MAX_PORT} {resources_args} --disable-usage-stats --include-dashboard=False"
    
    print(f"Starting Ray with command: {ray_start_command}")
    os.system(ray_start_command)

    init_hf_modules()  # Needed to avoid import error: https://github.com/vllm-project/vllm/pull/871
    ray.init(ignore_reinit_error=True)
    
