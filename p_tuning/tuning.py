from datasets import load_dataset, Dataset

from demo_watermark import load_model
from peft import get_peft_model, PromptEncoderConfig, TaskType
from transformers import Trainer, TrainingArguments, default_data_collator

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
            num_virtual_tokens=16,  # 对比{8, 16, 32}
            encoder_hidden_size=model.config.hidden_size,
        )

        model = get_peft_model(model, peft_config)
    return tokenizer, model


def train(model, tokenized_dataset, tokenizer):
    print("Start finetuning...")
    training_args = TrainingArguments(
        output_dir="./ptuned_opt",
        per_device_train_batch_size=4,  # 一个batch是一次送进模型的数据量
        gradient_accumulation_steps=4,  # 模型完成一次参数更新，有效 batch = 16 → 每个 epoch ≈ 1600/16 = 100 步
        num_train_epochs=5,  # 把数据集完整训练一次的轮数，小数据多跑几轮更稳

        # ---- P-tuning 常用较高 LR；小数据加 warmup 与 weight decay 抑制过拟合 ----
        learning_rate=8e-4,  # 可在 [5e-4, 2e-3] 网格微调
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # ---- 记录/保存：步数级评估，密集记录，避免“空图” ----
        # logging_strategy="steps",
        logging_steps=5,  # 100 步/epoch → 每轮约 20 个点
        logging_first_step=True,
        # evaluation_strategy="steps",
        eval_steps=20,  # 100 步/epoch → 每轮评估 5 次
        # save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,  # 需要提供 eval_dataset 才有效
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        fp16=True,
        seed=42,
        report_to=[],  # 不上报外部 logger
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    print("Finished!")

    logs = trainer.state.log_history
    df = pd.DataFrame(logs)
    loss_df = df[df["loss"].notna()].sort_values("step")
    loss_df["smoothed_loss"] = loss_df["loss"].rolling(20).mean()

    # 画图
    plt.figure(figsize=(6, 4))
    plt.plot(loss_df["step"], loss_df["smoothed_loss"])
    plt.xlabel("Global step")
    plt.ylabel("Training loss")
    plt.title("Training loss over steps")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_loss_curve.png", dpi=200)
    plt.close()


def get_ptuned_opt(args):
    dataset = load_training_data()
    tokenizer, model = load_configured_model(args)
    tokenized_dataset = dataset['train'].map(lambda x: tokenize_fn(x, tokenizer), remove_columns=["input", "target"])

    train(model, tokenized_dataset, tokenizer)

    # 保存 Prefix 参数
    model.save_pretrained("./ptuned_opt")
    tokenizer.save_pretrained("./ptuned_opt")

