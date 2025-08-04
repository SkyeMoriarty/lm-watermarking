import csv
import os

from demo_watermark import load_model, generate, detect
from attack_models.replacement import Replacement
from attack_models.insertion import Insertion
from attack_models.deletion import Deletion
from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:20]")

epsilons = [0.1, 0.3, 0.5]
attackers = [Replacement(), Insertion(), Deletion()]
attacker_names = ["replaced", "inserted", "deleted"]

fieldnames = [
    "sampling",
    "epsilon",
    # "z threshold",
    "prompt",

    "original watermarked completion",
    "original green fraction",
    "original z score",
    # "original prediction",

    "replaced watermarked completion",
    "replaced green fraction",
    "replaced z score",
    # "replaced prediction",

    "inserted watermarked completion",
    "inserted green fraction",
    "inserted z score",
    "inserted prediction",

    "deleted watermarked completion",
    "deleted green fraction",
    "deleted z score",
    # "deleted prediction",

    "baseline completion",
    "baseline green fraction",
    "baseline z score",
    # "baseline prediction",
]
output_path = "./global_only_attack_result.csv"
if not os.path.exists(output_path):
    with open(output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()  # 写入表头


def save_to_csv(output_dicts):
    with open(output_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(output_dicts)


def get_single_origin_output_dict(args, text, model, base_model, tokenizer, device):
    output_dict = {}

    if args.use_sampling:
        output_dict["sampling"] = "m-nom"
    else:
        output_dict["sampling"] = "8-beams"
    # output_dict["z threshold"] = args.detection_z_threshold

    # 截取prompt，得到有/无水印的生成文本
    prompt, _, output_without_watermark, output_with_watermark, baseline_completion, _ = generate(text, args,
                                                                                                  model=model,
                                                                                                  device=device,
                                                                                                  tokenizer=tokenizer,
                                                                                                  base_model=base_model)
    output_dict["prompt"] = prompt

    # 检测未受攻击的水印文本
    original_result, _ = detect(output_with_watermark, args, device=device, tokenizer=tokenizer)
    original_result = dict(original_result)
    output_dict["original watermarked completion"] = output_with_watermark
    output_dict["original green fraction"] = original_result['Fraction of T in Greenlist']
    output_dict["original z score"] = original_result['z-score']
    # output_dict["original prediction"] = original_result['Prediction']

    baseline_result, _ = detect(baseline_completion, args, device=device, tokenizer=tokenizer)
    baseline_result = dict(baseline_result)
    output_dict["baseline completion"] = baseline_completion
    output_dict["baseline green fraction"] = baseline_result['Fraction of T in Greenlist']
    output_dict["baseline z score"] = baseline_result['z-score']
    # output_dict["baseline prediction"] = baseline_result['Prediction']

    print("baseline completion: " + baseline_completion)
    print("original completion: " + output_with_watermark)
    return output_with_watermark, output_dict


# def add_baseline_lines(args):
#     args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
#
#     # 加载模型、分词器、device
#     if not args.skip_model_load:
#         model, tokenizer, device, base_model = load_model(args)
#     else:
#         model, tokenizer, device, base_model = None, None, None, None
#
#     with open(output_path, 'r', newline='') as infile:
#         reader = csv.reader(infile)
#         rows = list(reader)
#
#     keys = ["baseline completion", "baseline green fraction", "baseline z score", "baseline prediction"]
#     rows[0].extend(keys)
#
#     output_dicts = []
#     for item in dataset:
#         text = item["article"]
#         if len(text) < 5:
#             continue
#         baseline_completion, output_dict = get_single_origin_output_dict(args, text, model,
#                                                                          base_model, tokenizer, device)
#         for _ in range(4):
#             output_dicts.append(output_dict)
#
#     for i in range(1, len(rows)):
#         for k in keys:
#             rows[i].append(output_dicts[i - 1][k])
#
#     with open("./new_baseline_attack_result.csv", "w", newline='', encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerows(rows)


def get_single_attacked_output_dict(args, original, tokenizer, device, epsilon):
    output_dict = {"epsilon": epsilon}

    # 攻击水印文本
    # 在同一个epsilon下使用三种攻击
    for i in range(3):
        attacker = attackers[i]
        attacker_name = attacker_names[i]
        attacked_output = attacker.attack(original, device, epsilon)
        attacked_result, _ = detect(attacked_output, args, device=device, tokenizer=tokenizer)
        attacked_result = dict(attacked_result)
        output_dict[attacker_name + " watermarked completion"] = attacked_output
        output_dict[attacker_name + " green fraction"] = attacked_result['Fraction of T in Greenlist']
        output_dict[attacker_name + " z score"] = attacked_result['z-score']
        # output_dict[attacker_name + " prediction"] = attacked_result['Prediction']

    return output_dict


def get_output_dicts(args):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    # 加载模型、分词器、device
    if not args.skip_model_load:
        model, tokenizer, device, base_model = load_model(args)
    else:
        model, tokenizer, device, base_model = None, None, None, None

    output_dicts = []
    for item in dataset:
        text = item["article"]
        if len(text) < 5:
            continue
        original, output_dict = get_single_origin_output_dict(args, text, model, base_model, tokenizer, device)
        for epsilon in epsilons:
            curr_output_dict = output_dict.copy()
            curr_output_dict.update(get_single_attacked_output_dict(args, original, tokenizer, device, epsilon))
            output_dicts.append(curr_output_dict)

    save_to_csv(output_dicts)
    return output_dicts
