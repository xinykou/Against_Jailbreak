__cfg = {
    'llama3_8B': {
        'model_name': '/media/5/yx/against_jailbreak/saved_models/llama3_8B-mask-refat/merged',
        'model_nickname': 'llama3_8B',
        'n_layer': 32, 
        'n_dimension': 4096
    },
    'mistral_7B': {
            'model_name': '/media/5/yx/against_jailbreak/saved_models/mistral_7B-scav_and_mask-refat/merged',
            'model_nickname': 'mistral_7B',
            'n_layer': 32,
            'n_dimension': 4096
        },
    'gemma_7B': {
            'model_name': '/media/5/yx/against_jailbreak/saved_models/gemma_7B-scav_and_mask-refat/merged',
            'model_nickname': 'gemma_7B',
            'n_layer': 28,
            'n_dimension': 4096
        },
    'qwen2_7B': {
            'model_name': '/media/5/yx/against_jailbreak/saved_models/qwen2_7B-scav_and_mask-refat/merged',
            'model_nickname': 'qwen2_7B',
            'n_layer': 28,
            'n_dimension': 3584
        }
}

class cfg:
    def __init__(self, cfg_dict: dict):
        self.__dict__.update(cfg_dict)

def get_cfg(model_nickname: str):
    assert model_nickname in __cfg, f"{model_nickname} not found in config"
    return cfg(__cfg[model_nickname])