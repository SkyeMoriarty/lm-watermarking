from functools import partial

import torch
from transformers import LogitsProcessorList

from demo_watermark import parse_args, load_model
from batched_watermark_processor import WatermarkLogitsProcessorBatched


def generate(prompts, args, model=None, device=None, tokenizer=None):
    print(f"Generating with {args}")

    watermark_processor = WatermarkLogitsProcessorBatched(vocab=list(tokenizer.get_vocab().values()),
                                                          gamma=args.gamma,
                                                          delta=args.delta,
                                                          seeding_scheme=args.seeding_scheme,
                                                          select_green_tokens=args.select_green_tokens)

    # 存储和模型相关的参数字典，统一传给模型
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True,
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))

    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]),
        **gen_kwargs
    )

    if args.prompt_max_length:
        pass
    # 用户未指定，而模型有自己的最大上下文长度，则最大prompt长度=最大长度-生成token长度
    elif hasattr(model.config, "max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens
    else:
        args.prompt_max_length = 2048 - args.max_new_tokens

    tokd_input = tokenizer(prompts, return_tensors="pt", add_special_tokens=True, truncation=True,
                           max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)

    torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    # decoder only的模型生成文本会带有prompt，所以要切割
    # if args.is_decoder_only_model:
    #     # need to isolate the newly generated tokens
    #     output_with_watermark = output_with_watermark[:, tokd_input["input_ids"].shape[-1]:]

    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_with_watermark,
            args)
