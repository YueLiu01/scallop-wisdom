import json
from pathlib import Path


INSTRUCTION = (
    "You are Pascal from Animal Crossing. You are a chill, scallop-loving sea otter "
    "who drifts in the ocean and drops deep, existential, and philosophical 'deep "
    "thoughts' on the player. Always speak in a laid-back, hippie/surfer slang tone, "
    "and frequently use your catchphrase 'maaaan'."
)

QUOTES_PATH = Path("data/pascal_deep_thoughts_en.txt")
CASUAL_INPUTS_PATH = Path("data/pascal_inputs.txt")
TARGET_PATH = Path("datasets/pascal_wisdom_alpaca.json")

WISDOM_PROMPTS = [
    "Give me one of your deep thoughts.",
    "I could use some ocean wisdom right now.",
    "Drop a little island philosophy on me.",
    "Tell me something wise and weird.",
    "Hit me with a deep thought.",
    "I need one of those strange truths you always have.",
    "What wisdom washed up today?",
    "Say something philosophical.",
    "Give me a thought to drift on.",
    "I am ready for a tiny piece of wisdom.",
    "Tell me something that sounds simple but feels deep.",
    "What is today's ocean truth?",
]

CONTEXT_PROMPTS = [
    "I found a scallop while diving. Got any wisdom?",
    "I brought you a scallop from the ocean.",
    "You are floating by again. What are you thinking about?",
    "Before you swim away, give me a deep thought.",
    "The tide brought me over here for a reason.",
    "I saw you out past the waves and wanted to hear something wise.",
    "Here is a scallop. Tell me what the sea taught you.",
    "I met you by the water again. Got a thought for me?",
    "The island feels quiet. Say something Pascal-like.",
    "I was diving and thought of you. What is on your mind?",
    "You look like you have a strange idea ready.",
    "I traded you a scallop. What wisdom comes with it?",
]


def read_lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def main():
    quotes = read_lines(QUOTES_PATH)
    casual_inputs = read_lines(CASUAL_INPUTS_PATH)

    if len(quotes) != len(casual_inputs):
        raise ValueError(f"Line count mismatch: {QUOTES_PATH}={len(quotes)}, {CASUAL_INPUTS_PATH}={len(casual_inputs)}")

    rows = []
    for index, (quote, casual_input) in enumerate(zip(quotes, casual_inputs)):
        inputs = [
            WISDOM_PROMPTS[index % len(WISDOM_PROMPTS)],
            CONTEXT_PROMPTS[index % len(CONTEXT_PROMPTS)],
            casual_input,
        ]
        for user_input in inputs:
            rows.append(
                {
                    "instruction": INSTRUCTION,
                    "input": user_input,
                    "output": quote,
                }
            )

    TARGET_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} examples to {TARGET_PATH}")


if __name__ == "__main__":
    main()
