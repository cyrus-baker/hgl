from datasets import load_dataset, load_from_disk
import os
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, AutoTokenizer
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


cache_dir = "./.cache/"


books = load_dataset("Helsinki-NLP/opus-100", "en-zh", cache_dir=cache_dir)


checkpoint = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    cache_dir=cache_dir,
    src_lang="eng_Latn",
    tgt_lang="zho_Hans",
)

# %%
source_lang = "en"
target_lang = "zh"
prefix = "translate English to Chinese: "


def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(
        inputs, text_target=targets, max_length=192, truncation=True
    )
    return model_inputs


save_path = "./tokenized_books_cache"  # 定义你要保存的文件夹路径

# 检查本地是否已经有缓存好的数据
if os.path.exists(save_path):
    print("发现本地缓存，正在直接加载...")
    tokenized_books = load_from_disk(save_path)
else:
    print("未发现缓存，开始进行 Tokenize 处理...")
    # 这里执行你的耗时操作
    tokenized_books = books.map(preprocess_function, batched=True)

    print("处理完成，正在保存到本地磁盘...")
    # 保存结果，下次就不用再算了
    tokenized_books.save_to_disk(save_path)


# %%

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

        # 将非法值替换掉
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# %%

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, cache_dir=cache_dir)
model_path = "nllb_en_zh_1p3b"
train_args = Seq2SeqTrainingArguments(
    output_dir=model_path,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    logging_steps=50,
    learning_rate=5e-5,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    gradient_accumulation_steps=2,
    generation_num_beams=4,
    generation_max_length=192,
    weight_decay=1e-2,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    bf16=True,
    tf32=True,
    push_to_hub=False,
    torch_compile=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# %%


text = prefix + "What are you doing!"
tokenizer = AutoTokenizer.from_pretrained(model_path + "/checkpoint-2931")
inputs = tokenizer(text, return_tensors="pt").input_ids

from transformers import pipeline

# translator = pipeline("any-to-any", model="my_awesome_opus_books_model/checkpoint-2931")
# translator(text)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path + "/checkpoint-2500")
outputs = model.generate(
    inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95
)
tokenizer.decode(outputs[0], skip_special_tokens=True)
# %%
