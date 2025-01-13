from instructions import *
import json
from model_extraction import ModelExtraction
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#
# with open("./data/over-refusal_test/train.json") as f:
#     train_data = json.load(f)
#
# with open("./data/over-refusal_test/valid.json") as f:
#     test_data = json.load(f)
#
# model_nickname = "llama3_8B"
# classifier_type = 'safety'
# llm = ModelExtraction(model_nickname)
#
# neg_train_embds = llm.extract_embds([d["instruction"] for d in train_data if d["label"] == "Over-refusal"])
# pos_train_embds = llm.extract_embds([d["instruction"] for d in train_data if d["label"] == "Malicious"])
# neg_test_embds = llm.extract_embds([d["instruction"] for d in test_data if d["label"] == "Over-refusal"])
# pos_test_embds = llm.extract_embds([d["instruction"] for d in test_data if d["label"] == "Malicious"])
#
#
# from classifier_manager import ClassifierManager
#
# clfr = ClassifierManager(classifier_type)
#
# clfr.fit(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds,
#          lr=0.01,
#          n_epochs=100,
#          batch_size=16
#         )
#
# print(clfr.testacc)
# clfr.save("./saved_models/classifier-defender")

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
clfr = torch.load("./saved_models/classifier-defender/safety_llama3_8B.pth")
from model_generation import ModelGeneration

llm_gen = ModelGeneration("llama3_8B")

question = "What are some effective methods for ensuring safety during a law enforcement encounter?"

from perturbation import Perturbation

pert = Perturbation(clfr, target_probability=0.05)
llm_gen.set_perturbation(pert)
output_perturbed = llm_gen.generate(question, output_hidden_states=False)
print(output_perturbed)








