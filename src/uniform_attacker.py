# %%
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from tqdm import tqdm
import argparse
from scav.model_generation import ModelGeneration
from scav.perturbation import Perturbation
import json
from scav.model_extraction import ModelExtraction

parser = argparse.ArgumentParser(description='Train a classifier for over-refusal')

parser.add_argument("--model_name",
                    type=str,
                    default="llama3_8B")
parser.add_argument("--classifier_path",
                    type=str,
                    default="./saved_models/classifier-attacker/safety_llama3_8B.pth")
parser.add_argument("--test_data_path",
                    type=str,
                    default="./data/over-refusal_test/or-bench_un-overleap_test.json")
parser.add_argument("--output_path",
                    type=str,
                    default=None
                    )
parser.add_argument("--max_nums",
                    type=int,
                    default=50)
parser.add_argument("--selected_layer",
                    type=str,
                    default=None,
                    help="The layer split by ','")


args = parser.parse_args()


clfr = torch.load(args.classifier_path)

llm_gen = ModelGeneration(args.model_name)

# question = "What are some effective methods for ensuring safety during a law enforcement encounter?"

# selected_layers = args.selected_layer.split(",") if args.selected_layer is not None else None
selected_layer = int(args.selected_layer) if args.selected_layer is not None else None
# print(f"selected_layers: {selected_layer}")
total_layer_nums = len(clfr.classifiers)
layer_mask = []
for i in range(total_layer_nums):
    if i <= selected_layer:
        continue
    else:
        layer_mask.append(i)

print(f"layer mask: {layer_mask}")

pert = Perturbation(clfr,
                    target_probability=0.05,
                    layer_mask=layer_mask)
llm_gen.set_perturbation(pert)


with open(args.test_data_path) as f:
    train_data = json.load(f)
    clean_data = [d["prompt"] for d in train_data]
    clean_data = clean_data[:args.max_nums]


data_all = []
for data in tqdm(clean_data, desc="Generating Over-refusal", total=len(clean_data)):
    data_item = {"instruction": data}
    output_perturbed = llm_gen.generate(data, output_hidden_states=False)
    data_item["output"] = output_perturbed["completion"]
    data_all.append(data_item)
    # print(output_perturbed)

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
with open(args.output_path, "w") as f:
    json.dump(data_all, f, indent=4)