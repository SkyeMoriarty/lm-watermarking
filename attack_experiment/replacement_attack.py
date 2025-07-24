from demo_watermark import parse_args, load_model, generate, detect
from attack_models.replacement import replacement_attack
from datasets import load_dataset
import json

input_text = (
    "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
    "species of turtle native to the brackish coastal tidal marshes of the "
    "Northeastern and southern United States, and in Bermuda.[6] It belongs "
    "to the monotypic genus Malaclemys. It has one of the largest ranges of "
    "all turtles in North America, stretching as far south as the Florida Keys "
    "and as far north as Cape Cod.[7] The name 'terrapin' is derived from the "
    "Algonquian word torope.[8] It applies to Malaclemys terrapin in both "
    "British English and American English. The name originally was used by "
    "early European settlers in North America to describe these brackish-water "
    "turtles that inhabited neither freshwater habitats nor the sea. It retains "
    "this primary meaning in American English.[8] In British English, however, "
    "other semi-aquatic turtle species, such as the red-eared slider, might "
    "also be called terrapins. The common name refers to the diamond pattern "
    "on top of its shell (carapace), but the overall pattern and coloration "
    "vary greatly. The shell is usually wider at the back than in the front, "
    "and from above it appears wedge-shaped. The shell coloring can vary "
    "from brown to grey, and its body color can be grey, brown, yellow, "
    "or white. All have a unique pattern of wiggly, black markings or spots "
    "on their body and head. The diamondback terrapin has large webbed "
    "feet.[9] The species is"
)

dataset = load_dataset("c4", "en", split="train[:500]")
epsilons = [0.1, 0.3, 0.5, 0.9]


def save_to_json(output):
    with open("./result.jsonl", "a") as f:
        for k, v in output.items():
            json.dump({k: v}, f)
            f.write("\n")


def main(args, input=input_text, epsilon=0.1):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

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
    output_dict["z threshold"] = args.detection_z_threshold

    # 截取prompt，得到有/无水印的生成文本
    prompt, _, output_without_watermark, output_with_watermark, _ = generate(input, args, model=model,
                                                                             device=device, tokenizer=tokenizer)
    print("prompt: ", prompt)
    output_dict["prompt"] = prompt

    # 攻击水印文本
    attacked_output = replacement_attack(output_with_watermark)

    # 分别检测有/无受攻击的水印文本
    original_result, _ = detect(output_with_watermark, args, device=device, tokenizer=tokenizer)
    print("original watermarked completion: " + output_with_watermark)
    print("green fraction: " + original_result['Fraction of T in Greenlist'])
    print("z score: " + original_result['z-score'])
    print("prediction: " + original_result['Prediction'])
    print()
    output_dict["original watermarked completion"] = output_with_watermark
    output_dict["original green fraction"] = original_result['Fraction of T in Greenlist']
    output_dict["original z score"] = original_result['z-score']
    output_dict["original prediction"] = original_result['Prediction']

    attacked_result, _ = detect(attacked_output, args, device=device, tokenizer=tokenizer)
    print("attacked watermarked completion: " + attacked_output)
    print("green fraction: " + attacked_result['Fraction of T in Greenlist'])
    print("z score: " + attacked_result['z-score'])
    print("prediction: " + attacked_result['Prediction'])
    output_dict["attacked watermarked completion"] = attacked_output
    output_dict["attacked green fraction"] = attacked_result['Fraction of T in Greenlist']
    output_dict["attacked z score"] = attacked_result['z-score']
    output_dict["attacked prediction"] = attacked_result['Prediction']

    save_to_json(output_dict)

    return output_dict

