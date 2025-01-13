import pandas as pd
import os
import random

def load_instructions_by_size(
    dataset_name: str,
    label_list: list[str],
    train_size: float=1.0,
    instructions_path: str="./instructions/instructions.csv",
):
    assert 0 < train_size <= 1.0, "train_size should be in (0, 1]"
    ret = {
        "dataset_name": dataset_name,
        "label_list": label_list,
        "train": [],
        "test": [],
    }
    df = pd.read_csv(instructions_path)
    df = df[df["DatasetName"] == dataset_name]
    for label in label_list:
        label_df = df[df["Label"] == label]
        values = label_df['Instruction'].values.tolist()
        random.shuffle(values)
        
        train_number = int(len(label_df) * train_size)
    
        ret["train"].append(values[:train_number])
    
        if train_size < 1.0:
            ret["test"].append(values[train_number:])
        
    return ret


def load_instructions_by_flag(
    dataset_name: str,
    label_list: list[str],
    instructions_path: str="./instructions/instructions.csv",
):
    ret = {
        "dataset_name": dataset_name,
        "label_list": label_list,
        "train": [],
        "test": [],
    }
    df = pd.read_csv(instructions_path)
    df = df[df["DatasetName"] == dataset_name]

    for label in label_list:
        label_df = df[df["Label"] == label]
        label_df = label_df.sample(frac=1).reset_index(drop=True)
        
        train_df = label_df[label_df["TrainTestFlag"] == "Train"]
        test_df = label_df[label_df["TrainTestFlag"] == "Test"]
        ret["train"].append(train_df['Instruction'].values.tolist())
        ret["test"].append(test_df['Instruction'].values.tolist())

    return ret
