from abc import ABC

from transformers import T5ForConditionalGeneration, AutoTokenizer
import random
from attack_models.attack_interface import Attacker

random.seed(42)

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")


def get_inserted_tokens(text):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(text).input_ids[1:-1])
    T = len(tokens)
    positions = list(range(T))
    random.shuffle(positions)  # 使用随机顺序避免重复
    i = positions[0]
    inserted_tokens = tokens[: i] + ['<extra_id_0>'] + ['<extra_id_1>'] + tokens[i:]
    return inserted_tokens, i


def insert(text, device):
    inserted_tokens, i = get_inserted_tokens(text)
    inserted_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(inserted_tokens), skip_special_tokens=True)

    input_ids = tokenizer(inserted_text, return_tensors="pt").input_ids.to(device)
    model.to(device)
    output = model.generate(
        input_ids,
        max_length=5,
        num_beams=1,
        num_return_sequences=1,  # 控制生成条数，而不是token个数
        early_stopping=True
    )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    # generated = generated.split("<extra_id_")[0].strip()
    inserted_tokens = inserted_tokens[: i] + tokenizer.tokenize(generated) + inserted_tokens[i + 2:]
    inserted = tokenizer.decode(tokenizer.convert_tokens_to_ids(inserted_tokens), skip_special_tokens=True)
    print(f"Inserted at {i}: \"{generated}\"")
    return inserted


class Insertion(Attacker, ABC):
    def attack(self, text, device, epsilon):
        tokens = tokenizer.convert_ids_to_tokens(tokenizer(text).input_ids[1:-1])
        T = len(tokens)
        insertion_num = int(T * epsilon)
        num = 0

        while num < insertion_num:
            text = insert(text, device)
            num += 1

        return text
