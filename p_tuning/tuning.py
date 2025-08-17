from datasets import load_dataset, Dataset

from demo_watermark import load_model
from peft import get_peft_model, PromptEncoderConfig, TaskType
from transformers import Trainer, default_data_collator, IntervalStrategy, SchedulerType
from transformers.training_args import TrainingArguments
from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt


def load_training_data(path='p_tuning/p_tuning_data.cleaned.jsonl'):
    dataset = load_dataset("json", data_files=path)
    return dataset


def tokenize_fn(single, tokenizer):
    prompt = single["input"].strip()
    target = single["target"].strip()
    full = prompt + " " + target
    tokenized = tokenizer(full, padding="max_length", truncation=True, max_length=256)

    # attention mask告诉模型哪些token需要被关注，padding位置上的mask就是0
    if isinstance(tokenized['attention_mask'][0], list):  # 如果是嵌套列表
        tokenized['attention_mask'] = tokenized['attention_mask'][0]  # 取第一个元素

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids) + 1  # +1 for the space we added

    # labels告诉模型训练中期望预测的token，供计算loss使用
    # 对于无需预测的token（如padding），label就会设为-100
    labels = tokenized["input_ids"][:]
    # mask out prompt tokens and padding
    for i in range(len(labels)):
        if i < prompt_len or tokenized["attention_mask"][i] == 0:
            labels[i] = -100
    tokenized["labels"] = labels
    return tokenized


def load_configured_model(args):
    if not args.skip_model_load:
        model, tokenizer, device, _ = load_model(args)
    else:
        model, tokenizer, device, _ = None, None, None, None

    if not args.skip_model_load:
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            num_virtual_tokens=8,  # 对比{8, 16, 32}
            encoder_hidden_size=model.config.hidden_size,
        )

        model = get_peft_model(model, peft_config)
    return tokenizer, model


def smooth(y, k=5):
    import numpy as np
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k)/k, mode="valid")


def train(model, tokenized_dataset, tokenizer):
    print("Start finetuning...")
    training_args = TrainingArguments(
        output_dir="./ptuned_opt",
        per_device_train_batch_size=4,  # batch：一次送进模型的样本，这里一个batch中有4个sample
        # 指跑完8个batch后做一次参数更新，前8次得到的梯度累积起来，模拟大batch的更新效果，有效 batch = 4*8=32个sample
        gradient_accumulation_steps=8,   # step：一次参数更新，gradient_accumulation_steps指累计多少batch更新一次参数
        num_train_epochs=3,  # 已知32个sample更新一次参数，一个epoch内更新3000/30=100次参数，即一个epoch有100个step
        learning_rate=1e-3,
        weight_decay=0.01,  # 权重衰减，在损失函数里额外加上一项，惩罚权重参数过大=>防止模型过拟合

        eval_steps=25,  # 一般希望每个epoch进行1-5次验证
        eval_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=10,  # 一般希望每个epoch打印5-20次日志，每10步打印一次 loss，共打印10次
        logging_first_step=True,
        logging_dir="./logs",
        logging_strategy=IntervalStrategy.STEPS,  # 明确按 step 记录

        save_steps=50,
        save_total_limit=2,  # 只保留最近 2 个 checkpoint，省磁盘

        fp16=True,
        warmup_ratio=0.1,  # 前10%步逐步提高学习率，防止初期不稳定
        max_grad_norm=1.0,  # 梯度裁剪，防止梯度爆炸
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    print("Finished!")

    logs = trainer.state.log_history
    train_pts, eval_pts = [], []

    for rec in logs:
        if "loss" in rec and "learning_rate" in rec:  # 训练步骤日志
            train_pts.append((rec.get("step", None), rec["loss"]))
        if "eval_loss" in rec:  # 评估日志
            eval_pts.append((rec.get("step", None), rec["eval_loss"]))

    # 分别取 x,y
    tx, ty = zip(*train_pts) if train_pts else ([], [])
    ex, ey = zip(*eval_pts) if eval_pts else ([], [])

    k = 7
    ty_smooth = smooth(list(ty), k)
    tx_smooth = list(tx)[:len(ty_smooth)]  # 对齐长度

    plt.figure(figsize=(7, 4.5))
    plt.plot(tx, ty, label="Train loss (raw)", alpha=0.3)
    plt.plot(tx_smooth, ty_smooth, label=f"Train loss (smoothed, k={k})", linewidth=2)
    plt.plot(ex, ey, label="Eval loss", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("P-tuning on OPT: Train vs Eval Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./P-tuning on OPT: Train vs Eval Loss.png")
    plt.show()


def get_ptuned_opt(args):
    dataset = load_training_data()
    tokenizer, model = load_configured_model(args)
    tokenized_dataset = dataset['train'].map(lambda x: tokenize_fn(x, tokenizer), remove_columns=["input", "target"])

    train(model, tokenized_dataset, tokenizer)

    # 保存 Prefix 参数
    model.save_pretrained("./ptuned_opt")
    tokenizer.save_pretrained("./ptuned_opt")

