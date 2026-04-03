import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW

def train_model(model, train_dataset, tokenizer, batch_size=8, num_epochs=1, device="cuda"):
    # 使用 HF 的 DataCollatorForLanguageModeling，mlm=False 表示进行 Causal LM 训练
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    loss_fct = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        total_loss = 0
        for step, batch in enumerate(progress_bar):
            # 将数据移动到设备上 (GPU/CPU)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            # 模型输出 shape: (batch_size, seq_len, vocab_size)
            logits = model(input_ids)
            
            # 准备计算 loss，将 logits 和 labels shift 一位
            # 因为我们在预测 next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算 Loss (展平后计算)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    return model
