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


def load_data():
    dataset = load_dataset("dbpedia_14", split="train")
    return dataset


def get_dataloader(dataset, batch_size=8):
    # collate_fn参数指定如何将一批样本打包成一个batch
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: [e["text"] for e in x])
    return dataloader


def get_train_data(args):
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)
    print()

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    train_data = []
    if not args.skip_model_load:
        dataset = load_data()
        dataloader = get_dataloader(dataset)
        for prompts in dataloader:  # 遍历batch
            _, _, decoded_output_with_watermark, _ = generate(prompts, args, model=model,
                                                              device=device, tokenizer=tokenizer)
            # print(decoded_output_with_watermark)
            # print()
            train_data.append(decoded_output_with_watermark)
        print("len: " + str(len(train_data)))
    return train_data


# def format_train_data(train_data):


if __name__ == "__main__":
    args = parse_args()
    print(args)

    get_train_data(args)
