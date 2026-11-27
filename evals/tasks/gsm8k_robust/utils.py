import re

BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
HASHES = re.compile(r"####\s*([^\n]*)")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str:
    """Final numeric answer: last \\boxed{...}, else the text after the last '####', else the last number."""
    for pattern in (BOXED, HASHES):
        found = pattern.findall(text)
        if found:
            numbers = NUMBER.findall(found[-1])
            if numbers:
                return numbers[-1]
    numbers = NUMBER.findall(text)
    return numbers[-1] if numbers else "[invalid]"


def extract_answers(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_answer(r) for r in resp] for resp in resps]
