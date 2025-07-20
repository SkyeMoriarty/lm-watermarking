from transformers import T5ForConditionalGeneration, AutoTokenizer
import random

random.seed(42)

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")


def get_inserted_tokens(text):
    tokens = tokenizer.tokenize(text)
    i = random.randint(0, len(tokens))
    inserted_tokens = tokens[: i] + ['<extra_id_0>'] + ['<extra_id_1>'] + tokens[i:]
    return inserted_tokens, i


def insert(text):
    inserted_tokens, i = get_inserted_tokens(text)
    inserted_text = tokenizer.convert_tokens_to_string(inserted_tokens)

    input_ids = tokenizer(inserted_text, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(
        input_ids,
        max_length=5,  # replacement的长度
        num_beams=1,
        num_return_sequences=1,  # replacement的个数
        early_stopping=True
    )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    inserted_tokens = inserted_tokens[: i] + generated + inserted_tokens[i+2:]
    inserted = tokenizer.convert_tokens_to_string(inserted_tokens)
    return inserted


def insertion_attack(text, epsilon=0.1):
    tokens = tokenizer.tokenize(text)
    T = len(tokens)
    replacement_num = int(T * epsilon)
    num = 0

    while num < replacement_num:
        text = insert(text)
        num += 1

    return text
