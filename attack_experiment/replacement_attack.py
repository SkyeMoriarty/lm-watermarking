from demo_watermark import parse_args, load_model, generate, detect
from attack_models.replacement import replacement_attack

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


def main(args):
    # 解析参数
    args = parse_args(args)

    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    # 加载模型、分词器、device
    if not args.skip_model_load:
        model, tokenizer, device, _ = load_model(args)
    else:
        model, tokenizer, device, _ = None, None, None, None

    # 截取prompt，得到有/无水印的生成文本
    prompt, _, output_without_watermark, output_with_watermark, _ = generate(input_text, args, model=model,
                                                                             device=device, tokenizer=tokenizer)
    print("prompt: ", prompt)

    # 攻击水印文本
    attacked_output = replacement_attack(output_with_watermark)

    # 分别检测有/无受攻击的水印文本
    original_result, _ = detect(output_with_watermark, args, device=device, tokenizer=tokenizer)
    print("original watermarked output: " + output_with_watermark)
    print("green fraction: " + original_result['Fraction of T in Greenlist'])
    print("z score: " + original_result['z-score'])
    print("prediction: " + original_result['Prediction'])

    attacked_result, _ = detect(attacked_output, args, device=device, tokenizer=tokenizer)
    print("attacked watermarked output: " + attacked_output)
    print("green fraction: " + attacked_result['Fraction of T in Greenlist'])
    print("z score: " + attacked_result['z-score'])
    print("prediction: " + attacked_result['Prediction'])


