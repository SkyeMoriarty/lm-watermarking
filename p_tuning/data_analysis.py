from collections import Counter
import re

import pandas as pd
import matplotlib.pyplot as plt


patterns = {
    "HTML tags": r"<[^>]+>",
    "Escapes": r"\\[ntr]",  # 转义字符
    "URLs": r"https?://\S+|www\.\S+",
    "Unicode specials": r"[^\x00-\x7F]"  # Unicode特殊字符
}


def get_length_distribution(df):
    df["prompt_len"] = df["input"].apply(lambda x: len(str(x).split()))
    df["target_len"] = df["target"].apply(lambda x: len(str(x).split()))

    # ===== Prompt =====
    plt.figure(figsize=(6, 4))
    plt.hist(df["prompt_len"], bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Prompt Length (words)")
    plt.ylabel("Count")
    plt.title("Prompt Length Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('./Prompt Length Distribution.png')
    plt.show()

    # ===== Target  =====
    plt.figure(figsize=(6, 4))
    plt.hist(df["target_len"], bins=30, color='salmon', edgecolor='black')
    plt.xlabel("Completion Length (words)")
    plt.ylabel("Count")
    plt.title("Completion Length Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('./Completion Length Distribution.png')
    plt.show()


def get_repeat_ratio(text, n=3):
    tokens = text.split()
    if len(tokens) < n:
        return 0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    repeat_ratio = 1 - len(set(ngrams)) / len(ngrams)
    return repeat_ratio


def draw_repeat_ratio(df):
    df["target_high_rep_3gram"] = df["target"].apply(lambda x: get_repeat_ratio(x, n=3))

    plt.figure(figsize=(6, 4))
    plt.hist(df["target_high_rep_3gram"], bins=30, edgecolor='black')
    plt.xlabel("repetitive 3gram ratio")
    plt.ylabel("Count")
    plt.title("Repetitive 3gram Ratio Count")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('./Repetitive 3gram Ratio Count.png')
    plt.show()


def special_char_stats(texts):
    joined_text = " ".join(texts)
    total_chars = len(joined_text)
    counts = {name: len(re.findall(pattern, joined_text))
              for name, pattern in patterns.items()}
    percents = {name: f"{count / total_chars:.6f}" for name, count in counts.items()}
    return percents


if __name__ == '__main__':
    df = pd.read_json('./p_tuning_data.jsonl', lines=True)
    percents = special_char_stats(df["target"].tolist())
    print(percents)
    # get_length_distribution(df)
    # draw_repeat_ratio(df)
