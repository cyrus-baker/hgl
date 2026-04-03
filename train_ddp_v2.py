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
from torch.utils.tensorboard import SummaryWriter
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


def reduce_mean(tensor):
    tensor = tensor.detach().clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def get_eval_split(lm_datasets):
    for split_name in ["test", "validation", "val"]:
        if split_name in lm_datasets:
            return split_name, lm_datasets[split_name]
    return None, None


@torch.inference_mode()
def evaluate(model, eval_dataloader, loss_fct, device, vocab_size, use_bf16):
    model.eval()

    loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
    token_count = torch.zeros(1, device=device, dtype=torch.float64)

    for batch in eval_dataloader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        key_padding_mask = None
        if "attention_mask" in batch:
            key_padding_mask = (batch["attention_mask"] == 0).to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(input_ids, key_padding_mask=key_padding_mask)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
            )

        valid_tokens = (shift_labels != -100).sum()
        loss_sum += loss.detach().to(torch.float64) * valid_tokens.to(torch.float64)
        token_count += valid_tokens.to(torch.float64)

    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    avg_loss = (loss_sum / token_count.clamp_min(1)).item()
    model.train()
    return avg_loss


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

    # 新增：周期性评估与 TensorBoard
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default=".cache/tensorboard")
    parser.add_argument("--max_eval_samples", type=int, default=1000)

    args = parser.parse_args()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", cache_dir="./.cache")
    lm_datasets = load_from_disk(".cache/lm_datasets")

    vocab_size = len(tokenizer)
    maxlen = 1024
    embed_dim = 256
    num_heads = 8
    feed_forward_dim = 256
    num_transformer_blocks = 8

    train_dataset = (
        lm_datasets["train"] # .select(range(10000 * 8))
        if len(lm_datasets["train"]) > 10000
        else lm_datasets["train"]
    )

    eval_split_name, eval_dataset = get_eval_split(lm_datasets)
    if eval_dataset is not None and args.max_eval_samples > 0:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=True,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_sampler = DistributedSampler(
            eval_dataset,
            shuffle=False,
            drop_last=False,
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_batch_size,
            sampler=eval_sampler,
            shuffle=False,
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
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    use_bf16 = True
    global_step = 0

    writer = None
    if is_main_process():
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
        if eval_dataloader is None:
            print("警告：没有找到 test/validation/val split，因此不会进行测试集 loss 评估。")

    model.train()
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)

        if is_main_process():
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        else:
            progress_bar = train_dataloader

        epoch_loss_sum = 0.0
        epoch_step_count = 0

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

            global_step += 1
            reduced_loss = reduce_mean(loss)
            reduced_loss_value = reduced_loss.item()

            epoch_loss_sum += reduced_loss_value
            epoch_step_count += 1

            if is_main_process():
                progress_bar.set_postfix({"loss": f"{reduced_loss_value:.4f}"})

                if writer is not None and global_step % args.logging_steps == 0:
                    writer.add_scalar("train/loss_step", reduced_loss_value, global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            # 每隔一段时间测一次测试集 loss
            if eval_dataloader is not None and global_step % args.eval_steps == 0:
                eval_loss = evaluate(
                    model=model,
                    eval_dataloader=eval_dataloader,
                    loss_fct=loss_fct,
                    device=device,
                    vocab_size=vocab_size,
                    use_bf16=use_bf16,
                )

                if is_main_process():
                    print(f"\n[Eval] step={global_step}, {eval_split_name}_loss={eval_loss:.4f}")
                    if writer is not None:
                        writer.add_scalar(f"{eval_split_name}/loss_step", eval_loss, global_step)

        avg_train_loss = epoch_loss_sum / max(epoch_step_count, 1)

        if is_main_process():
            print(f"Epoch {epoch + 1} Average Train Loss: {avg_train_loss:.4f}")
            if writer is not None:
                writer.add_scalar("train/loss_epoch", avg_train_loss, epoch + 1)

        # 每个 epoch 结束也测一次
        if eval_dataloader is not None:
            eval_loss = evaluate(
                model=model,
                eval_dataloader=eval_dataloader,
                loss_fct=loss_fct,
                device=device,
                vocab_size=vocab_size,
                use_bf16=use_bf16,
            )

            if is_main_process():
                print(f"[Eval] epoch={epoch + 1}, {eval_split_name}_loss={eval_loss:.4f}")
                if writer is not None:
                    writer.add_scalar(f"{eval_split_name}/loss_epoch", eval_loss, epoch + 1)

    if is_main_process():
        os.makedirs(".cache", exist_ok=True)
        torch.save(model.module.state_dict(), ".cache/minigpt_ddp.pth")
        print("训练结束，主进程已保存模型。")
        if writer is not None:
            writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()