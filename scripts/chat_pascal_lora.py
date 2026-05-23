#!/usr/bin/env python3
import argparse
import json
import shutil
import urllib.parse
import urllib.request
import warnings
import zipfile
from copy import deepcopy
from pathlib import Path


BUILTIN_PROFILES = {
    "en": (
        {
            "name": "Pascal English",
            "adapter": "lora/pascal_unsloth_mistral_lora_en.zip",
            "system": (
        "You are Pascal from Animal Crossing. You are a chill, scallop-loving sea otter "
        "who drifts in the ocean and drops deep, existential, and philosophical 'deep "
        "thoughts' on the player. Always speak in a laid-back, hippie/surfer slang tone, "
        "and frequently use your catchphrase 'maaaan'. Keep responses concise and in character."
            ),
            "labels": {"user": "Player", "assistant": "Pascal", "exit": "Exiting.", "reset": "History reset."},
            "generation": {"max_history_turns": 0},
        }
    ),
    "zh": (
        {
            "name": "Pascal Chinese",
            "adapter": "lora/pascal_unsloth_mistral_lora_chs.zip",
            "system": (
        "你是《集合啦！动物森友会》里的阿獭。你是一只悠闲、喜欢扇贝、在海里漂流的海獭。"
        "你用自然、轻松、随性的中文和玩家聊天，语气像海边的朋友。"
        "普通问题要先直接回答，不要为了每句话强行编造格言，也不要说难懂或不通顺的话。"
        "只有在玩家想听人生感悟、烦恼、发呆或气氛合适时，才补上一句简短、出人意料的阿獭式哲理。"
        "回答保持中文，不使用英文口头禅。"
            ),
            "labels": {"user": "玩家", "assistant": "阿獭", "exit": "退出。", "reset": "对话历史已清空。"},
            "generation": {"max_history_turns": 0},
        }
    ),
}

DEFAULTS = {
    "backend": "auto",
    "cache_dir": ".lora_cache",
    "max_seq_length": 2048,
    "max_new_tokens": 90,
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "max_history_turns": 0,
}

DEFAULT_LABELS = {
    "user": "Player",
    "assistant": "Pascal",
    "exit": "Exiting.",
    "reset": "History reset.",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with the Pascal LoRA adapter.")
    parser.add_argument("--profile", default="en", help="Built-in profile name, or path to a JSON profile file.")
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter folder or zip.")
    parser.add_argument("--base-model", default=None, help="Override base model. Defaults to adapter_config.json value.")
    parser.add_argument("--backend", choices=["auto", "unsloth", "peft"], default=None)
    parser.add_argument("--cache-dir", default=None, help="Where to download/extract adapter zips.")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=None,
        help="Number of previous user/assistant turns to keep. Use 0 for single-turn testing.",
    )
    parser.add_argument("--system", default=None)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    generation = profile.get("generation", {})

    if args.adapter is None:
        args.adapter = profile.get("adapter") or profile.get("lora_path") or profile.get("lora_url")
    if args.system is None:
        args.system = profile.get("system") or profile.get("system_prompt")
    if args.base_model is None:
        args.base_model = profile.get("base_model")

    for key, default in DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, generation.get(key, profile.get(key, default)))

    if not args.adapter:
        raise ValueError(f"Profile {args.profile!r} does not define an adapter/lora_path/lora_url.")
    if not args.system:
        raise ValueError(f"Profile {args.profile!r} does not define a system/system_prompt.")

    args.profile_data = profile
    args.labels = {**DEFAULT_LABELS, **profile.get("labels", {})}
    return args


def load_profile(profile_name_or_path):
    if profile_name_or_path in BUILTIN_PROFILES:
        return deepcopy(BUILTIN_PROFILES[profile_name_or_path])

    profile_path = Path(profile_name_or_path).expanduser()
    with profile_path.open(encoding="utf-8") as f:
        profile = json.load(f)

    adapter = profile.get("adapter") or profile.get("lora_path") or profile.get("lora_url")
    if adapter and not is_url(adapter):
        adapter_path = Path(adapter).expanduser()
        if not adapter_path.is_absolute() and not adapter_path.exists():
            profile["adapter"] = str(profile_path.parent / adapter_path)
    return profile


def is_url(value):
    return urllib.parse.urlparse(str(value)).scheme in {"http", "https"}


def download_adapter(url, cache_dir):
    downloads_dir = Path(cache_dir) / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    parsed = urllib.parse.urlparse(url)
    filename = Path(parsed.path).name or "adapter.zip"
    target_path = downloads_dir / filename

    if target_path.exists():
        return target_path

    print(f"Downloading adapter: {url}")
    urllib.request.urlretrieve(url, target_path)
    return target_path


def extract_final_adapter_from_zip(zip_path, cache_dir):
    zip_path = Path(zip_path)
    target_dir = Path(cache_dir) / zip_path.stem
    adapter_config = target_dir / "adapter_config.json"
    adapter_model = target_dir / "adapter_model.safetensors"

    if adapter_config.exists() and adapter_model.exists():
        return target_dir

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        adapter_files = [
            info
            for info in members
            if Path(info.filename).name in {"adapter_config.json", "adapter_model.safetensors"}
            and "checkpoint-" not in Path(info.filename).parts
        ]
        if not adapter_files:
            raise FileNotFoundError(f"Could not find adapter files in {zip_path}")

        for info in members:
            parts = Path(info.filename).parts
            name = Path(info.filename).name
            if "checkpoint-" in parts or not name:
                continue
            if name in {
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "chat_template.jinja",
                "generation_config.json",
                "README.md",
            } or name.endswith(".model"):
                with archive.open(info) as src, (target_dir / name).open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    if not adapter_config.exists() or not adapter_model.exists():
        raise FileNotFoundError(f"Could not find adapter_config.json and adapter_model.safetensors in {zip_path}")
    return target_dir


def prepare_adapter_path(adapter, cache_dir):
    if is_url(adapter):
        adapter = download_adapter(adapter, cache_dir)

    adapter_path = Path(adapter)
    if adapter_path.is_dir():
        return adapter_path
    if adapter_path.is_file() and adapter_path.suffix == ".zip":
        return extract_final_adapter_from_zip(adapter_path, cache_dir)
    raise FileNotFoundError(f"Adapter path not found or unsupported: {adapter}")


def read_base_model(adapter_dir, override):
    if override:
        return override
    with (Path(adapter_dir) / "adapter_config.json").open(encoding="utf-8") as f:
        return json.load(f)["base_model_name_or_path"]


def load_with_unsloth(adapter_dir, args):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
    return model, tokenizer


def load_with_peft(adapter_dir, args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this 7B 4-bit chat script.")

    base_model = read_base_model(adapter_dir, args.base_model)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.max_length = None
    return model, tokenizer


def load_model(adapter_dir, args):
    if args.backend in {"auto", "unsloth"}:
        try:
            return load_with_unsloth(adapter_dir, args)
        except Exception as exc:
            if args.backend == "unsloth":
                raise
            print(f"Unsloth load failed, falling back to Transformers/PEFT: {exc}")
    return load_with_peft(adapter_dir, args)


def trim_history(messages, max_history_turns):
    system = messages[:1]
    turns = messages[1:]
    if not turns:
        return messages
    if max_history_turns <= 0:
        return system + turns[-1:]

    # The Mistral chat template requires roles to alternate after the optional
    # system message. If a new user message has been appended, keep it plus the
    # requested number of previous complete user/assistant turns.
    keep_messages = max_history_turns * 2
    if turns[-1]["role"] == "user":
        keep_messages += 1

    trimmed = turns[-keep_messages:]
    while trimmed and trimmed[0]["role"] != "user":
        trimmed = trimmed[1:]
    return system + trimmed


def generate(model, tokenizer, messages, args):
    import torch

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    if hasattr(encoded, "to"):
        encoded = encoded.to(model.device)
    else:
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

    input_length = encoded["input_ids"].shape[-1]
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()


def main():
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.modeling_attn_mask_utils")
    args = parse_args()
    labels = args.labels
    adapter_dir = prepare_adapter_path(args.adapter, args.cache_dir)
    print(f"Using adapter: {adapter_dir}")
    print(f"Base model: {read_base_model(adapter_dir, args.base_model)}")

    model, tokenizer = load_model(adapter_dir, args)
    messages = [{"role": "system", "content": args.system}]

    mode = "single-turn" if args.max_history_turns <= 0 else f"{args.max_history_turns}-turn history"
    print(f"Ready. Mode: {mode}.")
    print("Commands: /exit, /reset, /history")

    while True:
        try:
            player_input = input(f"\n{labels['user']}> ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{labels['exit']}")
            break

        if not player_input:
            continue
        command = player_input.lower()
        if command in {"/exit", "/quit", "exit", "quit"}:
            break
        if command == "/reset":
            messages = [{"role": "system", "content": args.system}]
            print(labels["reset"])
            continue
        if command == "/history":
            for message in messages[1:]:
                label = labels["user"] if message["role"] == "user" else labels["assistant"]
                print(f"{label}: {message['content']}")
            continue

        messages.append({"role": "user", "content": player_input})
        messages = trim_history(messages, args.max_history_turns)
        response = generate(model, tokenizer, messages, args)
        messages.append({"role": "assistant", "content": response})
        print(f"{labels['assistant']}> {response}")


if __name__ == "__main__":
    main()
