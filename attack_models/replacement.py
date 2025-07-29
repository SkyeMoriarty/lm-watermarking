from abc import ABC

from transformers import T5ForConditionalGeneration, AutoTokenizer
import random

from attack_models.attack_interface import Attacker

random.seed(42)

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-large")  # T5的预训练任务之一：span filling
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-large")


# 【随机】选择一个词用<extra_id_0> and <extra_id_1>包围——可复现！
# 返回原词和masked tokens
def get_masked_tokens(tokens, i):
    origin = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[i]), skip_special_tokens=True).strip()
    masked = tokens[: i] + ['<extra_id_0>'] + ['<extra_id_1>'] + tokens[i+1:]
    return origin, masked


# 用T5模型得到k=20个candidate替换词
# 根据和原词是否一致判断攻击是否成功
def is_successful(tokens, device, i, k=20):
    origin, masked = get_masked_tokens(tokens, i)
    masked_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(masked), skip_special_tokens=True)

    input_ids = tokenizer(masked_text, return_tensors="pt").input_ids.to(device)
    model.to(device)
    outputs = model.generate(
        input_ids,
        max_length=5,  # replacement的长度
        num_beams=1,
        num_return_sequences=k,  # replacement的个数
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        early_stopping=True,
    )

    decoded = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    for candidate in decoded:
        if candidate.strip().lower() != origin.strip().lower():
            replaced_tokens = masked[: i] + tokenizer.tokenize(candidate) + masked[i+2:]
            # print(f"Replaced token {i}: '{origin}' → '{candidate}'")
            return True, replaced_tokens
    return False, None


class Replacement(Attacker, ABC):
    def attack(self, text, device, epsilon, max_attempts=100):
        # 确保和替换词时的分词粒度一致
        tokens = tokenizer.convert_ids_to_tokens(tokenizer(text).input_ids[1:-1])
        T = len(tokens)
        replacement_num = int(T*epsilon)

        positions = list(range(T))
        random.shuffle(positions)  # 使用随机顺序避免重复
        num = 0
        attempt = 0

        for i in positions:
            attempt += 1
            if attempt > max_attempts:
                break
            if num >= replacement_num:
                break
            success, replaced_tokens = is_successful(tokens, device, i)
            if success:
                num += 1
                tokens = replaced_tokens

        text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)
        print(f"When epsilon={epsilon}, replaced text: ", text)

        return text

