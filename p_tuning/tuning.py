from datasets import load_dataset, Dataset

from demo_watermark import load_model
from peft import get_peft_model, PromptEncoderConfig, TaskType
from transformers import Trainer, TrainingArguments, default_data_collator

import pandas as pd
import matplotlib.pyplot as plt


def load_training_data(path='./p_tuning_data.jsonl'):
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
        per_device_train_batch_size=4,
        num_train_epochs=2,
        learning_rate=5e-5,
        save_total_limit=1,
        logging_steps=10,  # 每隔多少step记录一次
        save_steps=500,
        logging_dir="./logs",
        fp16=True,
        label_names=["labels"]
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

    # 画图
    plt.figure(figsize=(6, 4))
    plt.plot(loss_df["step"], loss_df["loss"])
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

