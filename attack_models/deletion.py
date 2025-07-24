import random
import spacy
from transformers import AutoTokenizer

from attack_models.attack_interface import Attacker

nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
random.seed(42)


class Deletion(Attacker):
    def attack(self, text, device, epsilon=0.1):
        doc = nlp(text)
        tokens = [token.text for token in doc]
        pos_tags = [token.pos_ for token in doc]

        deletable_indices = [i for i, pos in enumerate(pos_tags)
                             if pos in ['ADV', 'ADJ', 'DET', 'PART', 'INTJ', 'CCONJ']]
        if not deletable_indices:
            return text

        num_to_delete = int(len(tokens) * epsilon)
        indices_to_delete = random.sample(deletable_indices, min(num_to_delete, len(deletable_indices)))

        new_tokens = [tok for i, tok in enumerate(tokens) if i not in indices_to_delete]
        new_text = " ".join(new_tokens)

        # token_ids = tokenizer(new_text, return_tensors="pt").input_ids
        # return tokenizer.decode(token_ids[0], skip_special_tokens=True)
        return new_text
