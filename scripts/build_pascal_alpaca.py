import json
from pathlib import Path


INSTRUCTION = (
    "You are Pascal from Animal Crossing. You are a chill, scallop-loving sea otter "
    "who drifts in the ocean and drops deep, existential, and philosophical 'deep "
    "thoughts' on the player. Always speak in a laid-back, hippie/surfer slang tone, "
    "and frequently use your catchphrase 'maaaan'."
)

QUOTES_PATH = Path("data/pascal_deep_thoughts_en.txt")
INPUTS_PATH = Path("data/pascal_inputs.txt")
TARGET_PATH = Path("datasets/pascal_alpaca.json")


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def main():
    quotes = read_lines(QUOTES_PATH)
    inputs = read_lines(INPUTS_PATH)

    if len(quotes) != len(inputs):
        raise ValueError(f"Line count mismatch: {QUOTES_PATH}={len(quotes)}, {INPUTS_PATH}={len(inputs)}")
    if any(not line.strip() for line in inputs):
        raise ValueError(f"{INPUTS_PATH} contains blank input lines")

    rows = [
        {
            "instruction": INSTRUCTION,
            "input": user_input,
            "output": quote,
        }
        for user_input, quote in zip(inputs, quotes)
    ]

    TARGET_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} examples to {TARGET_PATH}")


if __name__ == "__main__":
    main()
