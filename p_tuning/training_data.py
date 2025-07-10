"""
    Get training data set for P-tuning v2, using the watermarked system.
    load model =>
    load dataset =>
    cut out prompts =>
    generate decoded_output_with_watermark =>
    concatenate
"""

from demo_watermark import parse_args, load_model, generate
from torchtext.datasets import WikiText2


def load_data():
    train_data = WikiText2(split='valid')
    train_texts = list(train_data)
    return train_texts


def truncate_prompt(tokenizer, texts, max_tokens=64):
    prompts = []
    for text in texts:
        tokenized = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors="pt")
        prompt_ids = tokenized["input_ids"][0].tolist()  # 0是取batch的第一维
        print("prompt_ids: " + prompt_ids)
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
        data = load_data()
        prompts = truncate_prompt(tokenizer, data)
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
