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
            num_virtual_tokens=16,  # 对比{8, 16, 32}
            encoder_hidden_size=model.config.hidden_size,
        )

        model = get_peft_model(model, peft_config)
    return tokenizer, model


def train(model, tokenized_dataset, tokenizer):
    print("Start finetuning...")
    training_args = TrainingArguments(
        output_dir="./ptuned_opt",
        per_device_train_batch_size=4,  # 显存吃紧时：2 或 1
        gradient_accumulation_steps=8,  # 有效 batch = 4*8=32
        num_train_epochs=3,
        learning_rate=5e-3,
        weight_decay=0.01,  # 权重衰减，在损失函数里额外加上一项，惩罚权重参数过大=>防止模型过拟合

        logging_steps=5,  # 每 5 步（每更新5次参数）打印一次 loss
        logging_first_step=True,
        logging_dir="./logs",
        logging_strategy=IntervalStrategy.STEPS,  # 明确按 step 记录

        save_steps=50,  # 每 50 步保存一次checkpoint
        save_total_limit=2,  # 只保留最近 2 个 checkpoint，省磁盘

        fp16=True,
        warmup_ratio=0.1,  # 前10%步逐步提高学习率，防止初期不稳定
        max_grad_norm=1.0,  # 梯度裁剪，防止梯度爆炸
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

