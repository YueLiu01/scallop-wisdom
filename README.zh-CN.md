# Pascal / 阿獭 LoRA

[English](README.md) | [中文](README.zh-CN.md)

这是一个给 Pascal / 阿獭做 LoRA 微调的小项目。阿獭是《动物森友会》里那只在海上漂来漂去、爱讲人生小道理的海獭。本仓库包含英文和简体中文语录、Alpaca 格式训练集、Colab 训练 notebook，以及本地测试 LoRA 的聊天脚本。

目标很简单：给模型一点扇贝大小的推力，让它更容易说出阿獭那种奇妙的小真理。

大的 LoRA 文件不会提交到 Git。默认聊天 profile 会直接从 Hugging Face 加载公开 LoRA adapter；如果你想保留本地副本，也可以放在 `lora/` 里，这个目录已经被 Git 忽略。

## 仓库内容

```text
data/
  pascal_deep_thoughts_en.txt     英文语录，一行一句
  pascal_deep_thoughts_zh.txt     简体中文语录，一行一句
  pascal_inputs.txt               与英文语录逐行对应的英文 casual inputs

datasets/
  pascal_alpaca.json              英文宽泛单轮训练集
  pascal_wisdom_alpaca.json       英文“请求智慧”训练集
  pascal_alpaca_chs.json          简体中文单轮训练集

profiles/
  pascal_en.json                  英文聊天 profile
  pascal_zh.json                  中文聊天 profile

scripts/
  build_pascal_alpaca.py          重新生成 datasets/pascal_alpaca.json
  build_pascal_wisdom_alpaca.py   重新生成 datasets/pascal_wisdom_alpaca.json
  chat_pascal_lora.py             本地 LoRA 聊天测试脚本

lora/                             可选的本地 LoRA zip/目录，Git 会忽略
```

## LoRA Adapters

训练好的 Mistral LoRA adapters 已经公开上传到 Hugging Face：

- 英文 Pascal：https://huggingface.co/CasperYL/pascal-unsloth-mistral-lora-en
- 简体中文阿獭：https://huggingface.co/CasperYL/pascal-unsloth-mistral-lora-chs

使用内置 profile 时，不需要手动下载。聊天脚本可以直接读取这些 adapter repo IDs。

如果你想测试自己下载的 zip，也可以把它放进 `lora/`，然后用 `--adapter` 指定。

## 本地聊天测试

先安装本地推理依赖。在仓库根目录运行：

```bash
conda create -n pascal-lora python=3.11 -y
conda activate pascal-lora
```

根据你的 NVIDIA 驱动安装 CUDA 版 PyTorch。如果 `nvidia-smi` 显示 CUDA 12.x，通常可以使用：

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

然后安装项目依赖：

```bash
pip install -U -r requirements.txt
```

可选：如果想用 Unsloth backend：

```bash
pip install -U -r requirements-train.txt
```

检查 PyTorch 是否能看到 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

英文 Pascal：

```bash
python scripts/chat_pascal_lora.py --profile en
```

中文阿獭：

```bash
python scripts/chat_pascal_lora.py --profile zh
```

聊天脚本默认是单轮模式，这更适合检查 LoRA 是否学到了语录风格。如果想保留多轮历史：

```bash
python scripts/chat_pascal_lora.py --profile zh --max-history-turns 6
```

聊天循环内支持：

```text
/history
/reset
/exit
```

## 示例输出

下面是默认公开 LoRA profiles 的单轮生成示例。生成时使用 sampling，所以你实际运行时的回答可能不同。

英文：

```text
Player> I found something shiny near the shore.
Pascal> It's hard to tell real gold from fool's gold. But if it's heavy, it's probably real.

Player> Why do snacks taste better outside?
Pascal> Maybe it's because food tastes better when you're eating it while ignoring your responsibilities, maaan...

Player> I dropped an apple in the sand.
Pascal> Apples are like the balls of fruit, maaan. They drop, then bounce. The others just sorta hang there.
```

简体中文：

```text
玩家> 今天阳光照在海面上很好看。
阿獭> 现在的太阳是过去的太阳的光。

玩家> 我坐在海边发呆了很久。
阿獭> 如果真实存在的地方就是现在，那么躺在沙滩上也是冒险

玩家> 岛上的大家都很忙，只有我在发呆。
阿獭> 有些人总是忙于一点也没做的工作
```

也可以覆盖 profile 里的 adapter：

```bash
python scripts/chat_pascal_lora.py \
  --profile zh \
  --adapter lora/pascal_unsloth_mistral_lora_chs.zip
```

## Profiles

Profile 是 JSON 文件，用来描述角色提示词、adapter、终端显示标签和生成参数。内置 profile 是 `en` 和 `zh`，也可以使用外部 profile 文件：

```bash
python scripts/chat_pascal_lora.py --profile profiles/pascal_zh.json
```

示例：

```json
{
  "name": "Pascal Chinese",
  "adapter": "CasperYL/pascal-unsloth-mistral-lora-chs",
  "system": "你是《集合啦！动物森友会》里的阿獭……",
  "labels": {
    "user": "玩家",
    "assistant": "阿獭",
    "exit": "退出。",
    "reset": "对话历史已清空。"
  },
  "generation": {
    "max_history_turns": 0,
    "max_new_tokens": 90,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.05
  }
}
```

`adapter` 可以是 Hugging Face repo ID、本地 adapter 目录、本地 `.zip`，也可以是一个指向 adapter zip 的 `https://...` 链接。命令行参数会覆盖 profile 里的值。

## 训练

推荐的 Colab notebooks：

```text
finetune_unsloth.ipynb       英文
finetune_unsloth_chs.ipynb   简体中文
```

在 Colab 中：

1. 打开 notebook。
2. 选择 `Runtime -> Change runtime type -> GPU`。
3. 运行 setup cell。
4. 上传对应的数据集 JSON：
   - `datasets/pascal_alpaca.json`
   - `datasets/pascal_wisdom_alpaca.json`
   - `datasets/pascal_alpaca_chs.json`
5. 开始训练。
6. 运行保存和测试 cells。
7. zip 并下载保存好的 LoRA adapter。

Unsloth notebooks 默认使用 Mistral：

```python
model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
```

你可以换成其他 Unsloth instruct model，但要注意：不同 base model 的风格模仿和聊天行为可能差别很大。

## 数据集说明

Alpaca 记录大概长这样：

```json
{
  "instruction": "You are Pascal from Animal Crossing...",
  "input": "The clouds looked weird today.",
  "output": "You ever look up at the clouds in the sky and imagine shapes? They're doin' the same thing to you, maaan."
}
```

数据集用途：

- `datasets/pascal_alpaca.json`：英文宽泛单轮 prompts。
- `datasets/pascal_wisdom_alpaca.json`：英文“给我一句智慧”这种更窄的行为。
- `datasets/pascal_alpaca_chs.json`：简体中文阿獭单轮 prompts。

目前中文 LoRA 更适合作为单轮语录/风格生成器，而不是长多轮聊天机器人。如果想让它有更强的连续对话能力，需要额外构造多轮聊天数据。

重新生成英文数据集：

```bash
python scripts/build_pascal_alpaca.py
python scripts/build_pascal_wisdom_alpaca.py
```

## 更小的 LoRA Zip

Colab 导出的 zip 可能包含 checkpoint 目录和 optimizer states。推理时只需要最终 adapter 文件。如果你导出的是 adapter 目录，可以先删除 checkpoints 再打包：

```bash
rm -rf lora/pascal_unsloth_mistral_lora_en/checkpoint-*
zip -r lora/pascal_unsloth_mistral_lora_en_small.zip lora/pascal_unsloth_mistral_lora_en
```

聊天脚本在解压 adapter zip 时也会跳过 checkpoint 目录。

## License / Rights Note

这是一个粉丝向研究项目，用于个人实验和角色风格微调测试。本项目与 Nintendo、Animal Crossing 或任何官方权利方无关。Animal Crossing 和阿獭属于其相应权利方。

请同时遵守 base model 以及已托管 LoRA adapters 的 license 和使用条款。

## TODO

- 改善多轮聊天质量。目前 LoRA 在单轮模式下表现更好；开启历史记录后，回答可能会跑偏或重复，因为当前训练数据主要是单轮样本。

## 最后一条小想法

原始语录像小石子。Alpaca 数据像贴了标签的小石子。LoRA 就像潮水反复摸过这些石子以后，模型终于觉得它们本来就该在那里。
