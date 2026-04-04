import torch
from lt import MiniGPT
from transformers import AutoTokenizer


maxlen = 256

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", cache_dir="./.cache")

vocab_size = len(tokenizer)
num_transformer_blocks = 8
maxlen = 1024
embed_dim = 256
num_heads = 8
feed_forward_dim = 256
batch_size = 144 * 1 / 2  # divide by 2 in case of model parallelism


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(
        max_len=maxlen,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=feed_forward_dim,
        num_transformer_blocks=num_transformer_blocks
    ).to(device)

    sd = torch.load(".cache/minigpt_ddp.pth")
    model.load_state_dict(sd)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=True):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print('model size: {:.3f}MB'.format(size_all_mb))

    print(model)

    start_text = "What is you name? Anna says: \"My name is "
    start_token = tokenizer.encode(start_text)
    model.generate_text(256, start_token, tokenizer, device)