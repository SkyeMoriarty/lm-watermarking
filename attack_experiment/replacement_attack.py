import csv
import os

from demo_watermark import parse_args, load_model, generate, detect
from attack_models.replacement import replacement_attack
from datasets import load_dataset


dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:5]")

epsilons = [0.1, 0.3, 0.5, 0.9]

fieldnames = [
    "sampling",
    "epsilon",
    "z threshold",
    "prompt",
    "original watermarked completion",
    "original green fraction",
    "original z score",
    "original prediction",
    "attacked watermarked completion",
    "attacked green fraction",
    "attacked z score",
    "attacked prediction",
]
output_path = "./result.csv"
if not os.path.exists(output_path):
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头


def save_to_csv(output_dicts):
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(output_dicts)


def get_single_output_dict(args, input, epsilon=0.1):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    # 加载模型、分词器、device
    if not args.skip_model_load:
        model, tokenizer, device, _ = load_model(args)
    else:
        model, tokenizer, device, _ = None, None, None, None

    output_dict = {}

    if args.use_sampling:
        output_dict["sampling"] = "m-nom"
    else:
        output_dict["sampling"] = "8-beams"
    output_dict["epsilon"] = epsilon
    output_dict["z threshold"] = args.detection_z_threshold

    # 截取prompt，得到有/无水印的生成文本
    prompt, _, output_without_watermark, output_with_watermark, _ = generate(input, args, model=model,
                                                                             device=device, tokenizer=tokenizer)
    output_dict["prompt"] = prompt

    # 攻击水印文本
    attacked_output = replacement_attack(output_with_watermark, device)

    # 分别检测有/无受攻击的水印文本
    original_result, _ = detect(output_with_watermark, args, device=device, tokenizer=tokenizer)
    original_result = dict(original_result)
    # print("original watermarked completion: " + output_with_watermark)
    # print("green fraction: " + original_result['Fraction of T in Greenlist'])
    # print("z score: " + original_result['z-score'])
    # print("prediction: " + original_result['Prediction'])
    # print()
    output_dict["original watermarked completion"] = output_with_watermark
    output_dict["original green fraction"] = original_result['Fraction of T in Greenlist']
    output_dict["original z score"] = original_result['z-score']
    output_dict["original prediction"] = original_result['Prediction']

    attacked_result, _ = detect(attacked_output, args, device=device, tokenizer=tokenizer)
    attacked_result = dict(attacked_result)
    # print("attacked watermarked completion: " + attacked_output)
    # print("green fraction: " + attacked_result['Fraction of T in Greenlist'])
    # print("z score: " + attacked_result['z-score'])
    # print("prediction: " + attacked_result['Prediction'])
    output_dict["attacked watermarked completion"] = attacked_output
    output_dict["attacked green fraction"] = attacked_result['Fraction of T in Greenlist']
    output_dict["attacked z score"] = attacked_result['z-score']
    output_dict["attacked prediction"] = attacked_result['Prediction']

    return output_dict


def get_output_dicts(args):
    output_dicts = {}

    for item in dataset:
        text = item["article"]
        print("text: " + text)
        for epsilon in epsilons:
            output_dicts.update(get_single_output_dict(args, text, epsilon))

    save_to_csv(output_dicts)
    return output_dicts
