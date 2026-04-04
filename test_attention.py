import torch

print("flash built:", torch.backends.cuda.is_flash_attention_available())

# 具体 SDPAParams 的构造签名，直接看你本机：
# help(torch.backends.cuda.SDPAParams)

params = torch.backends.cuda.SDPAParams(
    q, k, v, attn_mask, dropout_p, is_causal, False
)

print("can flash:", torch.backends.cuda.can_use_flash_attention(params, debug=True))
print("can efficient:", torch.backends.cuda.can_use_efficient_attention(params, debug=True))
print("can cudnn:", torch.backends.cuda.can_use_cudnn_attention(params, debug=True))