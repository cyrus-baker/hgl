from itertools import chain
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch
import os

print(os.environ.get("HF_ENDPOINT"))

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", cache_dir="./.cache")

# 测试一段多语言文本
text = "Hello world! 你好，世界！नमस्ते दुनिया! (Hindi)"
tokens = tokenizer.tokenize(text)
print(f"Qwen Tokens: {tokens}")
print(tokenizer.decode(tokenizer.encode(text)))

vocab_size = len(tokenizer)
num_transformer_blocks = 8
maxlen = 256
embed_dim = 256
num_heads = 8
feed_forward_dim = 256
batch_size = 144 * 1 / 2  # divide by 2 in case of model parallelism

num_epochs = 1
top_k = 10

num_proc = 32

data = load_from_disk("my_tinystory_local")

# ======== 开始 HF 常规数据处理 ========
def tokenize_function(examples):
    # 使用 tokenizer 将文本转为 token ID
    return tokenizer(examples["text"])

print("正在进行 Tokenization...")
if not os.path.exists(".cache/tokenized_datasets"):
    tokenized_datasets = data.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc, # 可以根据你的 CPU 核心数调整
        remove_columns=["text"], # 移除原始文本列，只保留 token ids
    )
    tokenized_datasets.save_to_disk(".cache/tokenized_datasets")
else:
    tokenized_datasets = load_from_disk(".cache/tokenized_datasets")




def group_texts(examples):
    # 在每段文本末尾加上 EOS token 隔离上下文
    eos_id = tokenizer.eos_token_id
    for i in range(len(examples["input_ids"])):
        examples["input_ids"][i].append(eos_id)
        if "attention_mask" in examples:
            examples["attention_mask"][i].append(1)

    # 使用 itertools.chain 极大地加速列表拼接
    concatenated_examples = {
        k: list(chain.from_iterable(examples[k]))
        for k in examples.keys()
    }
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= maxlen:
        total_length = (total_length // maxlen) * maxlen

    result = {
        k: [t[i : i + maxlen] for i in range(0, total_length, maxlen)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

print(f"正在将数据打包成 maxlen={maxlen} 的块...")
if not os.path.exists(".cache/lm_datasets"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
    )
    lm_datasets.save_to_disk(".cache/lm_datasets")
else:
    lm_datasets = load_from_disk(".cache/lm_datasets")





print(f"数据处理完成！训练集共有 {len(lm_datasets['train'])} 个样本。")
# ======== 结束 HF 常规数据处理 ========


def causal_attention_mask(seq_len):
    return torch.tril(torch.ones((seq_len, seq_len)))


class TransformerBlock(nn.Module):
    """A single Transformer block.

    Each Transformer block processes input sequences via self-attention and feed-forward networks.

    Args:
        embed_dim (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        ff_dim (int): Dimensionality of the feed-forward network.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
        rate (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        *,
        seed: int = 0,
        rate: float = 0.1,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(rate)

        self.layer_norm1 = nn.LayerNorm(
            normalized_shape=embed_dim,
            eps=1e-6,
        )

        self.linear1 = nn.Linear(
            embed_dim,
            ff_dim,
        )

        self.linear2 = nn.Linear(
            ff_dim,
            embed_dim,
        )

        self.dropout2 = nn.Dropout(rate)

        self.layer_norm2 = nn.LayerNorm(
            normalized_shape=embed_dim,
            eps=1e-6,
        )

    def forward(self, inputs, key_padding_mask=None):
        input_shape = inputs.shape
        batch_size, seq_len, _ = input_shape

        # Apply Multi-Head Attention with the causal attention mask.
        attention_output = self.mha(
            query=inputs,
            key=inputs,
            value=inputs,
            need_weights=False,
            key_padding_mask=key_padding_mask,
            attn_mask=nn.Transformer.generate_square_subsequent_mask(seq_len),
            is_causal=True,  
        )[0]

        # Apply the first dropout.
        attention_output = self.dropout1(attention_output)
        # Apply the first layer normalization.
        out1 = self.layer_norm1(inputs + attention_output)

        # The feed-forward network.
        # Apply the first linear transformation.
        ffn_output = self.linear1(out1)
        # Apply the ReLU activation with gelu
        ffn_output = nn.functional.gelu(ffn_output)
        # Apply the second linear transformation.
        ffn_output = self.linear2(ffn_output)
        # Apply the second dropout.
        ffn_output = self.dropout2(ffn_output)
        # Apply the second layer normalization and return the output of the Transformer block.
        return self.layer_norm2(out1 + ffn_output)


class TokenAndPositionEmbedding(nn.Module):
    """Combines token embeddings (words in an input sentence) with
    positional embeddings (the position of each word in a sentence).

    Args:
        maxlen (int): Matimum sequence length.
        vocal_size (int): Vocabulary size.
        embed_dim (int): Embedding dimensionality.
        rngs (flax.nnx.Rngs): A Flax NNX stream of JAX PRNG keys.
    """

    def __init__(
        self,
        maxlen: int,
        vocab_size: int,
        embed_dim: int,
    ):
        super().__init__()
        # Initialize token embeddings (using `flax.nnx.Embed`).
        # Each unique word has an embedding vector.
        self.token_emb = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        # Initialize positional embeddings (using `flax.nnx.Embed`).
        self.pos_emb = nn.Embedding(num_embeddings=maxlen, embedding_dim=embed_dim)

    # Takes a token sequence (integers) and returns the combined token and positional embeddings.
    def forward(self, x):
        # Generate a sequence of positions for the input tokens.
        seq_len = x.shape[1]
        positions = torch.arange(0, seq_len, device=x.device)[None, :]
        # Look up the positional embeddings for each position in the input sequence.
        position_embedding = self.pos_emb(positions)
        # Look up the token embeddings for each token in the input sequence.
        token_embedding = self.token_emb(x)
        # Combine token and positional embeddings.
        return token_embedding + position_embedding


class MiniGPT(nn.Module):
    def __init__(
        self,
        max_len: int,
        vocab_size: int,
        embed_dim,
        num_heads,
        ff_dim: int,
        num_transformer_blocks,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedding_layer = TokenAndPositionEmbedding(
            max_len, vocab_size=vocab_size, embed_dim=embed_dim
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                ff_dim,
            )
            for _ in range(num_transformer_blocks)
        ])

        self.output_layer = nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
        )

    def forward(self, inputs):
        x = self.embedding_layer(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        # Pass the output of the transformer blocks through the output layer,
        # and obtain logits for each token in the vocabulary (for next token prediction).
        outputs = self.output_layer(x)
        return outputs

    # def sample_from(self, logits):
    #     logits, indices = torch.topk(logits, top_k)
    #     slogits = torch.softmax(logits)

    #     return indices[torch.multinomial(slogits, 1)]

    # def generate_step(self, padded_tokens, sample_index):
    #     logits = self(padded_tokens)
    #     next_token = self.sample_from(logits[0][sample_index])
    #     return next_token

    # def generate_text(self, max_tokens, start_tokens):
    #     generated = []
    #     print(tokenizer.decode(start_tokens), flush=True, end="")
    #     for i in range(max_tokens):
    #         sample_index = len(start_tokens) + len(generated) - 1

    #         padded_tokens = torch.tensor(
    #             (
    #                 start_tokens
    #                 + generated
    #                 + [0] * (maxlen - len(start_tokens) - len(generated))
    #             )
    #         )[None, :]
    #         next_token = int(self.generate_step(padded_tokens, sample_index))
    #         if (
    #             next_token
    #             == tokenizer.encode(tokenizer.special_tokens_map["eos_token"])[0]
    #         ):
    #             break
    #         generated.append(next_token)
    #         # decode and print next_token
    #         print(tokenizer.decode([next_token]), flush=True, end="")
    #     return tokenizer.decode(start_tokens + generated)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling
    from tqdm import tqdm
    import torch.optim as optim

    # 1. 实例化模型
    print("正在实例化 MiniGPT 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(
        max_len=maxlen,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=feed_forward_dim,
        num_transformer_blocks=num_transformer_blocks
    ).to(device)

    # 2. 准备 DataLoader
    print("准备 DataLoader...")
    # DataCollatorForLanguageModeling 会自动帮你处理 padding 和创建 Causal LM 需要的 labels
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 我们只拿一小部分数据测试训练，如果你想全量跑，去掉 select
    train_dataset = lm_datasets["train"]#.select(range(10000)) if len(lm_datasets["train"]) > 10000 else lm_datasets["train"]

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=8,  # 你可以根据显存调整
        collate_fn=data_collator
    )

    # 3. 设置优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    loss_fct = nn.CrossEntropyLoss()

    # 4. 开始训练循环
    print(f"开始训练，使用设备: {device}")
    model.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training")

        total_loss = 0
        for step, batch in enumerate(progress_bar):
            # 将 batch 里的数据挪到 GPU/CPU
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # 如果有 key_padding_mask，可以传入 model，这里暂时简写不传入
            # key_padding_mask = (batch["attention_mask"] == 0).to(device)

            optimizer.zero_grad()

            # 前向传播 (Shape: batch_size, seq_len, vocab_size)
            logits = model(input_ids)

            # 由于是自回归预测 (预测下一个 token)，我们需要把 logits 向左移一位，labels 向右移一位
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 展平后计算 CrossEntropyLoss
            loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    print("训练结束！")
    torch.save(model.state_dict(), ".cache/minigpt.pth")
