from transformers import T5ForConditionalGeneration, AutoTokenizer
import random

random.seed(42)

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")  # T5就是做span replacement的
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

# test
# 【随机】选择一个词用<extra_id_0> and <extra_id_1>包围——可复现！
# 返回原词和masked tokens
def get_masked_tokens(text):
    tokens = tokenizer.tokenize(text)
    i = random.randint(0, len(tokens) - 1)
    orign = tokenizer.convert_tokens_to_string(tokens[i]).strip()
    masked = tokens[: i] + ['<extra_id_0>'] + ['<extra_id_1>'] + tokens[i+1:]
    return orign, masked, i


# 用T5模型得到k=20个candidate替换词
# 根据和原词是否一致判断攻击是否成功
def is_successful(text, k=20, num_beams=50):
    origin, masked, i = get_masked_tokens(text)
    masked_text = tokenizer.convert_tokens_to_string(masked)

    input_ids = tokenizer(masked_text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(
        input_ids,
        max_length=10,  # replacement的长度
        num_beams=num_beams,
        num_return_sequences=k,  # replacement的个数
        early_stopping=True
    )

    decoded = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    for candidate in decoded:
        if candidate != origin:
            replaced_tokens = masked[: i] + tokenizer.tokenize(candidate) + masked[i+2:]
            replaced = tokenizer.convert_tokens_to_string(replaced_tokens)
            return True, replaced
    return False, None


def replacement_attack(text, epsilon, max_attempts=50):
    tokens = tokenizer.tokenize(text)
    T = len(tokens)
    replacement_num = T*epsilon
    num = 0
    attempt = 0

    while num < replacement_num:
        attempt += 1
        if attempt > max_attempts:
            break
        success, replaced_text = is_successful(text)
        if success:
            num += 1
            text = replaced_text

    return text
