# Pascal / 阿獭 LoRA

[English](README.md) | [中文](README.zh-CN.md)

A small fine-tuning project for Pascal, the drifting sea otter philosopher from Animal Crossing. The repo has English and Simplified Chinese quote datasets, Alpaca-format training files, Colab notebooks, and a local chat script for testing LoRA adapters.

The goal is simple: give a model a scallop-sized nudge toward Pascal's odd little truths, maaan.

Large LoRA files are not committed. Put them in `lora/`, which is ignored by Git.

## What Is Here

```text
data/
  pascal_deep_thoughts_en.txt     English quotes, one per line
  pascal_deep_thoughts_zh.txt     Simplified Chinese quotes, one per line
  pascal_inputs.txt               English casual inputs aligned to English quotes

datasets/
  pascal_alpaca.json              English broad single-turn dataset
  pascal_wisdom_alpaca.json       English wisdom-trigger dataset
  pascal_alpaca_chs.json          Simplified Chinese single-turn dataset

profiles/
  pascal_en.json                  English chat profile
  pascal_zh.json                  Chinese chat profile

scripts/
  build_pascal_alpaca.py          Rebuilds datasets/pascal_alpaca.json
  build_pascal_wisdom_alpaca.py   Rebuilds datasets/pascal_wisdom_alpaca.json
  chat_pascal_lora.py             Local LoRA chat tester

lora/                             Local LoRA zips/folders, ignored by Git
```

## LoRA Downloads

Download the trained Mistral LoRA zips into `lora/`.

```bash
mkdir -p lora

# English Pascal Mistral LoRA
curl -L "TODO_ENGLISH_MISTRAL_LORA_ZIP_URL" \
  -o lora/pascal_unsloth_mistral_lora_en.zip

# Simplified Chinese 阿獭 Mistral LoRA
curl -L "TODO_CHINESE_MISTRAL_LORA_ZIP_URL" \
  -o lora/pascal_unsloth_mistral_lora_chs.zip
```

## Chat Locally

Install local inference dependencies first. From the repo root:

```bash
conda create -n pascal-lora python=3.11 -y
conda activate pascal-lora
```

Install CUDA PyTorch based on your NVIDIA driver. If `nvidia-smi` shows CUDA 12.x, this is usually fine:

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the project dependencies:

```bash
pip install -U -r requirements.txt
```

Optional, for the Unsloth backend:

```bash
pip install -U -r requirements-train.txt
```

Check that PyTorch sees your GPU:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

English Pascal:

```bash
python scripts/chat_pascal_lora.py --profile en
```

Chinese 阿獭:

```bash
python scripts/chat_pascal_lora.py --profile zh
```

The chat script defaults to single-turn mode, which is best for checking whether the LoRA learned the quote style. To keep rolling history:

```bash
python scripts/chat_pascal_lora.py --profile en --max-history-turns 6
```

Inside the chat loop:

```text
/history
/reset
/exit
```

You can override a profile's adapter:

```bash
python scripts/chat_pascal_lora.py \
  --profile en \
  --adapter lora/pascal_unsloth_mistral_lora_en.zip
```

## Profiles

Profiles are JSON files that describe the character prompt, adapter, labels, and generation defaults. Built-in profiles are `en` and `zh`, and external profile files work too:

```bash
python scripts/chat_pascal_lora.py --profile profiles/pascal_en.json
```

Example:

```json
{
  "name": "Pascal English",
  "adapter": "../lora/pascal_unsloth_mistral_lora_en.zip",
  "system": "You are Pascal from Animal Crossing...",
  "labels": {
    "user": "Player",
    "assistant": "Pascal",
    "exit": "Exiting.",
    "reset": "History reset."
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

`adapter` can be a local folder, a local `.zip`, or an `https://...` link to an adapter zip. Command-line flags override profile values.

## Training

Recommended Colab notebooks:

```text
finetune_unsloth.ipynb       English
finetune_unsloth_chs.ipynb   Simplified Chinese
```

In Colab:

1. Open the notebook.
2. Choose `Runtime -> Change runtime type -> GPU`.
3. Run the setup cell.
4. Upload the matching dataset JSON:
   - `datasets/pascal_alpaca.json`
   - `datasets/pascal_wisdom_alpaca.json`
   - `datasets/pascal_alpaca_chs.json`
5. Train.
6. Run the save/test cells.
7. Zip and download the saved LoRA adapter.

The Unsloth notebooks use Mistral by default:

```python
model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
```

You can swap in another Unsloth instruct model, but keep in mind that style imitation and chat behavior can change a lot across base models.

## Dataset Notes

The Alpaca records look like this:

```json
{
  "instruction": "You are Pascal from Animal Crossing...",
  "input": "The clouds looked weird today.",
  "output": "You ever look up at the clouds in the sky and imagine shapes? They're doin' the same thing to you, maaan."
}
```

Dataset roles:

- `datasets/pascal_alpaca.json`: broad English single-turn prompts.
- `datasets/pascal_wisdom_alpaca.json`: narrower English "give me wisdom" behavior.
- `datasets/pascal_alpaca_chs.json`: Simplified Chinese 阿獭 single-turn prompts.

The Chinese LoRA currently works better as a single-turn quote/style generator than as a long multi-turn chatbot. A longer chat dataset would be needed for stronger conversational memory.

Rebuild generated English datasets:

```bash
python scripts/build_pascal_alpaca.py
python scripts/build_pascal_wisdom_alpaca.py
```

## Smaller LoRA Zips

Colab zips may include checkpoint folders and optimizer states. For inference, you only need the final adapter files. If you exported an adapter folder, remove checkpoints before zipping:

```bash
rm -rf lora/pascal_unsloth_mistral_lora_en/checkpoint-*
zip -r lora/pascal_unsloth_mistral_lora_en_small.zip lora/pascal_unsloth_mistral_lora_en
```

The chat script also skips checkpoint folders when extracting adapter zips.

## TODO

- Improve multi-turn chat quality. Current LoRAs work best in single-turn mode; with conversation history enabled, responses can drift or become repetitive because the training data is mostly single-turn.

## Final Little Thought

Raw quotes are pebbles. Alpaca rows are pebbles with little labels. A LoRA is what happens when the tide keeps touching the same pebbles until the model starts thinking they belong there.
