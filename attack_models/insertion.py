from transformers import T5ForConditionalGeneration, AutoTokenizer
import random

random.seed(42)

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")


def get_inserted_tokens(text):


def insertion_attack(text, epsilon, max_attempts=50):
    