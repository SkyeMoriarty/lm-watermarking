"""
    Get training data set for P-tuning v2, using the watermarked system.
    load model =>
    load dataset (prompts)=>
    generate decoded_output_with_watermark =>
    concatenate them =>
    format
"""

from demo_watermark import parse_args, load_model, generate
from datasets import load_dataset
from torch.utils.data import DataLoader

import json


def load_data():
    dataset = load_dataset("ag_news", split="train")
    subset = dataset.select(range(500, 1000))
    return subset


def get_dataloader(dataset, batch_size=2):
    # collate_fn参数指定如何将一批样本打包成一个batch
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: [e["text"] for e in x])
    return dataloader


def save_to_json(inputs, targets):
    with open("./p_tuning_data.jsonl", "a") as f:
        for input, target in zip(inputs, targets):
            json.dump({"input": input, "target": target}, f)
            f.write("\n")


def get_train_data(args):
    if not args.skip_model_load:
        model, tokenizer, device, _ = load_model(args)
    else:
        model, tokenizer, device, _ = None, None, None, None

    inputs = []  # 存储list[str]的列表
    targets = []
    if not args.skip_model_load:
        dataset = load_data()
        dataloader = get_dataloader(dataset)
        for prompt in dataloader:
            redecoded_input, _, _, decoded_output_with_watermark, _, _ = generate(prompt, args, model=model,
                                                                                  device=device, tokenizer=tokenizer)
            inputs.append(redecoded_input)
            targets.append(decoded_output_with_watermark)
            print()
            print("Input: " + redecoded_input)
            print()
            print("Target: " + decoded_output_with_watermark)

    save_to_json(inputs, targets)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    get_train_data(args)
