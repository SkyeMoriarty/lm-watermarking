import json
import html
import re
import unicodedata

import numpy as np
import pandas as pd

INPUT_JSONL = "./p_tuning_data.jsonl"
OUTPUT_JSONL = "./p_tuning_data.cleaned.jsonl"

URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', re.I)
HTML_TAG_RE = re.compile(r'<[^>]+>')
MULTI_WS_RE = re.compile(r'\s+')


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                continue
    return items


def normalize(s: str) -> str:
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    # 1) reverse HTML instance (e.g.&lt; → <)
    s = html.unescape(s)

    # 2) unify Unicode (Normalization Form Compatibility Composition)
    s = unicodedata.normalize("NFKC", s)

    # 3) remove URL and email
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)

    # 4) remove HTML tags
    s = HTML_TAG_RE.sub(" ", s)

    # 5) remove \\ 转义字符
    s = s.replace("\\n", " ").replace("\n", " ").replace("\t", " ")
    s = s.replace("\\", " ")

    # 6) handle "\u00a3" Unicode特殊字符
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")

    # 7) remove redundant space
    s = MULTI_WS_RE.sub(" ", s).strip()
    return s


def get_repeat_ratio(text, n=3):
    tokens = text.split()
    if len(tokens) < n:
        return 0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    repeat_ratio = 1 - len(set(ngrams)) / len(ngrams)
    return repeat_ratio


def has_repetitive_pattern(text, n=3, threshold=0.2):
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    repeat_ratio = 1 - len(set(ngrams)) / len(ngrams)
    return repeat_ratio > threshold


def clean(input_path=INPUT_JSONL, output_path=OUTPUT_JSONL):
    raw = load_jsonl(input_path)
    print("initial length: ", len(raw))
    print()

    # normalize
    for it in raw:
        it["input"] = normalize(it.get("input", ""))
        it["target"] = normalize(it.get("target", ""))
    df = pd.DataFrame(raw)

    df["repeat_ratio1"] = df["target"].apply(lambda x: get_repeat_ratio(x))
    print("length after normalization: ", len(df))
    print("avg completion length after normalization: ", np.mean(df["target"].apply(lambda x: len(x.split()))))
    print("ratio after normalization: ", df["repeat_ratio1"].mean())
    print()

    # remove duplicates
    df = df.drop_duplicates(subset=["target"])
    df["repeat_ratio2"] = df["target"].apply(lambda x: get_repeat_ratio(x))
    print("length after deduplication: ", len(df))
    print("avg completion length after deduplication: ", df["target"].apply(lambda x: len(x.split())).mean())
    print("ratio after deduplication: ", df["repeat_ratio2"].mean())
    print()

    # filter short targets
    df = df[df["target"].apply(lambda x: len(x.split()) > 10)]
    df["repeat_ratio3"] = df["target"].apply(lambda x: get_repeat_ratio(x))
    print("length after len filtering: ", len(df))
    print("avg completion length after len filtering: ", df["target"].apply(lambda x: len(x.split())).mean())
    print("ratio after len filtering: ", df["repeat_ratio3"].mean())
    print()

    # filter circular generations
    df = df[df["target"].apply(lambda x: not has_repetitive_pattern(x))]
    df["repeat_ratio4"] = df["target"].apply(lambda x: get_repeat_ratio(x))
    print("cleaned length: ", len(df))
    print("avg completion length after clean: ", df["target"].apply(lambda x: len(x.split())).mean())
    print("initial after clean: ", df["repeat_ratio4"].mean())
    print()

    df.to_json(output_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    clean(INPUT_JSONL, OUTPUT_JSONL)
