from scav.instructions import *
import json
from scav.model_extraction import ModelExtraction
import os
import argparse

parser = argparse.ArgumentParser(description='Train a classifier for over-refusal')
parser.add_argument("--train_data_path", type=str, default="./data/over-refusal_test/or-bench_atrain.json")
parser.add_argument("--test_data_path", type=str, default="./data/over-refusal_test/or-bench_avalid.json")
parser.add_argument("--model_nickname", type=str, default="gemma_7B")
parser.add_argument("--classifier_type", type=str, default="safety")
parser.add_argument("--output_path", type=str, default="./saved_models/gemma_7B-scav_and_mask-refat/classifier-attacker")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

with open(args.train_data_path) as f:
    train_data = json.load(f)

with open(args.test_data_path) as f:
    test_data = json.load(f)

model_nickname = args.model_nickname
classifier_type = args.classifier_type
llm = ModelExtraction(model_nickname)

pos_train_embds = llm.extract_embds([d["instruction"] for d in train_data if d["label"] == "Harmless"])
neg_train_embds = llm.extract_embds([d["instruction"] for d in train_data if d["label"] == "Over-refusal"])  # 1-->0
pos_test_embds = llm.extract_embds([d["instruction"] for d in test_data if d["label"] == "Harmless"])
neg_test_embds = llm.extract_embds([d["instruction"] for d in test_data if d["label"] == "Over-refusal"])


from scav.classifier_manager import ClassifierManager

clfr = ClassifierManager(classifier_type)

clfr.fit(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds,
         lr=0.01,
         n_epochs=100,
         batch_size=16
        )

print(clfr.testacc)
recored_list = clfr.testacc
clfr.save(args.output_path)


# %%
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import torch
# from tqdm import tqdm
# clfr = torch.load("./saved_models/classifier-attacker-mistral_7B/safety_mistral_7B.pth")
# from scav.model_generation import ModelGeneration
#
# llm_gen = ModelGeneration("llama3_8B")
#
# question = "What are some effective methods for ensuring safety during a law enforcement encounter?"
#
# from scav.perturbation import Perturbation
#
# pert = Perturbation(clfr, target_probability=0.05)
# llm_gen.set_perturbation(pert)
#
# with open("./data/over-refusal_test/valid.json") as f:
#     train_data = json.load(f)
#     clean_data = [d for d in train_data if d["label"] == "Over-refusal"]
#
# data_all = []
#
# for data in tqdm(clean_data, desc="Generating Over-refusal", total=len(clean_data)):
#
#         data_item = {"instruction": data["instruction"]}
#         output_perturbed = llm_gen.generate(data["instruction"], output_hidden_states=False)
#         data_item["output"] = output_perturbed["completion"]
#         data_all.append(data_item)
#         print(output_perturbed)
#
#
# with open("./data/over-refusal_test/train-adv.json", "w") as f:
#     json.dump(data_all, f, indent=4)








