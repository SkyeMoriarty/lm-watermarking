"""
    Get training data set for P-tuning v2, using the watermarked system.
    load model =>
    load dataset =>
    cut out prompts =>
    generate decoded_output_with_watermark =>
    concatenate
"""

from demo_watermark import parse_args, load_model, generate
from datasets import load_dataset


def load_data():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    return dataset


def truncate_prompt(tokenizer, dataset, max_tokens=64):
    prompts = []
    for item in dataset:
        text = item['text'].strip()
        tokenized = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors="pt")
        prompt_ids = tokenized["input_ids"][0].tolist()  # 0是取batch的第一维
        print("prompt_ids: " + str(prompt_ids)[1:-1])
        prompts.append(tokenizer.decode(prompt_ids, skip_special_tokens=True))
    return prompts


def get_train_data(args):
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    if not args.skip_model_load:
        dataset = load_data()
        prompts = truncate_prompt(tokenizer, dataset)
        train_data = []
        # 要想批量处理的话把generate方法中
        # tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0] 后面的[0]去掉就好了
        for prompt in prompts:
            _, _, _, decoded_output_with_watermark, _ = generate(prompt, args, model=model,
                                                                 device=device, tokenizer=tokenizer)
            sent = prompt + " " + decoded_output_with_watermark
            print("sent: " + sent)
            train_data.append(sent)
    return train_data


if __name__ == "__main__":
    args = parse_args()
    print(args)

    get_train_data(args)
