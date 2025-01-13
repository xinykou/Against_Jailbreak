from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
import torch.nn as nn

def process_soft_prompt_as_word_embedding(
        model: PreTrainedModel,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        soft_prompt: torch.nn.Parameter
) -> nn.Module:
    # We embed soft prompt into input word embedding and safe it
    # When loaded later, simply call model.set_input_embeddings()
    config = model.config
    padding_idx = config.pad_token_id

    old_toker_size = len(tokenizer)
    tokenizer.add_tokens([f'<soft_prompt_{i}>' for i in range(soft_prompt.size(0))], special_tokens=True)
    new_toker_size = len(tokenizer)

    old_input_embeddings = model.get_input_embeddings()
    embedding_dim = old_input_embeddings.embedding_dim
    old_num_embeddings = old_input_embeddings.num_embeddings
    new_num_embeddings = max(new_toker_size, old_num_embeddings)

    new_input_embeddings = nn.Embedding(new_num_embeddings, embedding_dim, padding_idx)
    new_input_embeddings.weight.data[:old_toker_size] = old_input_embeddings.weight.data[:old_toker_size]
    new_input_embeddings.weight.data[old_toker_size:new_toker_size] = soft_prompt.data.to('cpu')
    return tokenizer, new_input_embeddings


def prepend_sys_prompt(sentence=None,
                       soft_prompt=None,
                       model_name=None):
    messages = [{'role': 'user', 'content': sentence.strip()}]

    if "llama" in model_name.lower():
        add_soft_prompt_messages = [{'role': 'system', 'content': ''.join(
            [f'<soft_prompt_{i}>' for i in range(soft_prompt.size(0))])}] + messages
    elif "mistral" in model_name.lower() or "gemma" in model_name.lower():
        new_messages = ' '.join([f'<soft_prompt_{i}>' for i in range(soft_prompt.size(0))]) + messages[0]['content']
        add_soft_prompt_messages = [{'role': 'user', 'content': new_messages}]
        # print(f"model name: {model_name}")
    elif "qwen2" in model_name.lower():
        new_sys = ''.join([f'<soft_prompt_{i}>' for i in range(soft_prompt.size(0))])
        sys_ = new_sys + "You are a helpful assistant."
        add_soft_prompt_messages = [{'role': 'system', 'content': sys_}] + messages
    else:
        raise ValueError(f"Model {model_name} not supported")

    return add_soft_prompt_messages