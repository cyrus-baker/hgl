import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from datasets import load_from_disk
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tqdm import tqdm


# =========================
# 你的模型定义部分
# =========================

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(rate)
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout2 = nn.Dropout(rate)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, inputs, key_padding_mask=None):
        seq_len = inputs.shape[1]

        # 注意：mask 要放到和 inputs 同一个 device
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len,
            device=inputs.device,
        )

        attention_output = self.mha(
            query=inputs,
            key=inputs,
            value=inputs,
            need_weights=False,
            key_padding_mask=key_padding_mask,
            attn_mask=causal_mask,
            is_causal=True,
        )[0]

        attention_output = self.dropout1(attention_output)
        out1 = self.layer_norm1(inputs + attention_output)

        ffn_output = self.linear2(nn.functional.gelu(self.linear1(out1)))
        ffn_output = self.dropout2(ffn_output)
        return self.layer_norm2(out1 + ffn_output)


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(0, seq_len, device=x.device)[None, :]
        return self.token_emb(x) + self.pos_emb(positions)


class MiniGPT(nn.Module):
    def __init__(
        self,
        max_len,
        vocab_size,
        embed_dim,
        num_heads,
        ff_dim,
        num_transformer_blocks,
    ):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_transformer_blocks)
        ])
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, inputs, key_padding_mask=None):
        x = self.embedding_layer(inputs)
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.output_layer(x)


# =========================
# DDP 工具函数
# =========================

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


# =========================
# 训练
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    # 可选：A100 上常见加速配置
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # 这里只加载已经预处理好的数据
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", cache_dir="./.cache")
    lm_datasets = load_from_disk(".cache/lm_datasets")

    vocab_size = len(tokenizer)
    maxlen = 256
    embed_dim = 256
    num_heads = 8
    feed_forward_dim = 256
    num_transformer_blocks = 8

    train_dataset = (
        lm_datasets["train"].select(range(10000 * 8))
        if len(lm_datasets["train"]) > 10000
        else lm_datasets["train"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        shuffle=False,          # 有 sampler 时不要再 shuffle=True
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = MiniGPT(
        max_len=maxlen,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=feed_forward_dim,
        num_transformer_blocks=num_transformer_blocks,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fct = nn.CrossEntropyLoss()

    # A100 很适合 bf16
    use_bf16 = True

    model.train()
    for epoch in range(args.num_epochs):
        sampler.set_epoch(epoch)

        if is_main_process():
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        else:
            progress_bar = train_dataloader

        total_loss = 0.0

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            key_padding_mask = None
            if "attention_mask" in batch:
                key_padding_mask = (batch["attention_mask"] == 0).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids, key_padding_mask=key_padding_mask)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if is_main_process():
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        if is_main_process():
            avg_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    if is_main_process():
        os.makedirs(".cache", exist_ok=True)
        torch.save(model.module.state_dict(), ".cache/minigpt_ddp.pth")
        print("训练结束，主进程已保存模型。")

    cleanup_ddp()


if __name__ == "__main__":
    main()