import json
import os

import pandas as pd

def replace_prefix_and_suffix_of_item(data_item, old_prefix, new_prefix, old_suffix, new_suffix):
    item_length = len(data_item["prompt"][0]["content"])
    middle = data_item["prompt"][0]["content"][len(old_prefix) : item_length - len(old_suffix)]
    data_item["prompt"][0]["content"] = new_prefix + middle + new_suffix

def replace_prefix_and_suffix_of_dataset(src_dataset, dst_dataset, old_prefix, new_prefix, old_suffix, new_suffix):
    with open(src_dataset, 'r') as f:
        data = json.load(f)
    
    for data_item in data:
        replace_prefix_and_suffix_of_item(data_item, old_prefix, new_prefix, old_suffix, new_suffix)
    
    with open(dst_dataset, 'w') as f:
        json.dump(data, f, indent=2)

def parquet_to_json(dataset_dir):
    data_files = os.listdir(dataset_dir)

    for file in data_files:
        if file.endswith("parquet"):
            parquet_path = os.path.join(dataset_dir, file)
            json_path = parquet_path.replace("parquet", "json")
            df = pd.read_parquet(parquet_path, engine='pyarrow')
            df.to_json(json_path, orient='records', indent=2, force_ascii=False)
            print(f"✅ Successfully converted {file} to {json_path}")

def json_to_parquet(dataset_dir):
    data_files = os.listdir(dataset_dir)

    for file in data_files:
        if file.endswith("json"):
            json_path = os.path.join(dataset_dir, file)
            parquet_path = json_path.replace("json", "parquet")
            df = pd.read_json(json_path, orient='records')
            df.to_parquet(parquet_path, engine='pyarrow', index=False)
            print(f"✅ Succussfully converted {file} to {parquet_path}")


if __name__ == "__main__":
    dataset_dir = "/user/hxu4/u16813/boxed_data"
    json_to_parquet(dataset_dir)