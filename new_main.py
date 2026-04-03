from datasets import load_dataset, load_from_disk
import os
import time
import evaluate
import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


cache_dir = "./.cache/"
checkpoint = "facebook/nllb-200-distilled-600M"

source_lang = "en"
target_lang = "zh"

src_lang_code = "eng_Latn"
tgt_lang_code = "zho_Hans"

save_path = "./tokenized_books_cache"
done_flag = save_path + ".done"


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


def wait_for_cache():
    while not (os.path.exists(save_path) and os.path.exists(done_flag)):
        time.sleep(3)


books = load_dataset("Helsinki-NLP/opus-100", "en-zh", cache_dir=cache_dir)

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    cache_dir=cache_dir,
    src_lang=src_lang_code,
    tgt_lang=tgt_lang_code,
)


def preprocess_function(examples):
    inputs = [example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=192,
        truncation=True,
    )
    return model_inputs


def get_tokenized_books():
    if os.path.exists(save_path) and os.path.exists(done_flag):
        print("发现本地缓存，正在直接加载...")
        return load_from_disk(save_path)

    if is_main_process():
        print("未发现缓存，开始进行 Tokenize 处理...")
        tokenized_books = books.map(preprocess_function, batched=True)

        print("处理完成，正在保存到本地磁盘...")
        tokenized_books.save_to_disk(save_path)

        with open(done_flag, "w", encoding="utf-8") as f:
            f.write("ok\n")

        return tokenized_books
    else:
        print(f"Rank {os.environ.get('RANK')} 等待主进程完成缓存...")
        wait_for_cache()
        print(f"Rank {os.environ.get('RANK')} 检测到缓存已完成，开始加载...")
        return load_from_disk(save_path)


metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
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


def main():
    tokenized_books = get_tokenized_books()

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, cache_dir=cache_dir)
    model_path = "nllb_en_zh_600m"

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
        report_to="none",
        ddp_find_unused_parameters=False,
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

    # trainer.train()

    final_path = os.path.join(model_path, "final")
    # trainer.save_model(final_path)

    if is_main_process():
        # tokenizer.save_pretrained(final_path)

        text = "What is your name? My name is Shengyong Li."

        infer_tokenizer = AutoTokenizer.from_pretrained(
            final_path,
            src_lang=src_lang_code,
            tgt_lang=tgt_lang_code,
        )
        infer_model = AutoModelForSeq2SeqLM.from_pretrained(final_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        infer_model = infer_model.to(device)
        infer_model.eval()

        inputs = infer_tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = infer_model.generate(
                **inputs,
                forced_bos_token_id=infer_tokenizer.convert_tokens_to_ids(tgt_lang_code),
                max_new_tokens=40,
                do_sample=True,
                top_k=30,
                top_p=0.95,
            )

        print(infer_tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()