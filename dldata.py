from datasets import load_dataset

# 1. 下载数据集并保存到指定路径
# name="zh-en" 是指中英平行语料，也可以是 "de-en" 等
dataset = load_dataset("roneneldan/TinyStories", cache_dir="./.cache")

# 2. 将处理好的数据保存到本地磁盘
dataset.save_to_disk("./my_tinystory_local")

print("下载并保存完成！")