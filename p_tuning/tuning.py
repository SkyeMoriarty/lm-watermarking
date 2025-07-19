from datasets import load_dataset
from demo_watermark import load_model
from peft import get_peft_model, PrefixTuningConfig, TaskType
from transformers import Trainer, TrainingArguments


def load_training_data(path='./p_tuning_data.jsonl'):
    dataset = load_dataset("json", data_files=path)
    return dataset


def tokenize_fn(single, tokenizer):
    full_input = single['input'] + ' ' + single['target']
    tokenized = tokenizer(full_input,
                          padding="max_length",
                          truncation=True,
                          max_length=256)
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized


def load_configured_model(args):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)
    print()

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    if not args.skip_model_load:
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,  # 表示我们要训练，不是推理
            num_virtual_tokens=16,  # 每层transformer前加16个virtual token
            encoder_hidden_size=model.config.hidden_size,
            prefix_projection=True,
        )

        model = get_peft_model(model, peft_config)  # 得到注入了前缀token的model
    return tokenizer, model


def train(model, tokenized_dataset):
    print("Start finetuning...")
    training_args = TrainingArguments(
        output_dir="./ptuned_opt",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        # 和logging相关的参数
        save_total_limit=1,
        logging_steps=10,
        save_steps=500,
        logging_dir="./logs",
        label_names=["labels"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    print("Finished!")


def get_ptuned_opt(args):
    dataset = load_training_data()
    tokenizer, model = load_configured_model(args)
    tokenized_dataset = dataset['train'].map(lambda x: tokenize_fn(x, tokenizer), remove_columns=["input", "target"])  # 移除原始列名
    train(model, tokenized_dataset)

    # 保存 Prefix 参数
    model.save_pretrained("./ptuned_opt")
    tokenizer.save_pretrained("./ptuned_opt")

