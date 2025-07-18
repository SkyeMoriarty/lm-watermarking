"""
    Get training data set for P-tuning v2, using the watermarked system.
    load model =>
    load dataset (prompts)=>
    generate decoded_output_with_watermark =>
    concatenate them =>
    format
"""

from demo_watermark import parse_args, load_model
from watermark import generate
from datasets import load_dataset
from torch.utils.data import DataLoader

import json


def load_data():
    dataset = load_dataset("ag_news", split="train")
    subset = dataset.select(range(10000))
    return subset


def get_dataloader(dataset, batch_size=2):
    # collate_fn参数指定如何将一批样本打包成一个batch
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: [e["text"] for e in x])
    return dataloader


def save_to_json(inputs, targets):
    with open("./p_tuning_data.jsonl", "w") as f:
        for input, target in zip(inputs, targets):
            json.dump({"input": input, "target": target}, f)
            f.write("\n")


def get_train_data(args):
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)
    print()

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    inputs = []  # 存储list[str]的列表
    targets = []
    if not args.skip_model_load:
        dataset = load_data()
        dataloader = get_dataloader(dataset)
        for prompts in dataloader:  # 遍历batch
            redecoded_input, _, decoded_output_with_watermark, _ = generate(prompts, args, model=model,
                                                                            device=device, tokenizer=tokenizer)
            for input_str, target_str in zip(redecoded_input, decoded_output_with_watermark):
                inputs.append(input_str)
                targets.append(target_str)
            print()
            print("Input: ")
            print(redecoded_input)
            print()
            print("Target: ")
            print(decoded_output_with_watermark)
            save_to_json(inputs, targets)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    get_train_data(args)
