# lm-eval's mbpp utils load the code-execution metric on import, which fails unless HF_ALLOW_CODE_EVAL=1,
# so import them lazily: only when this task actually runs.


def pass_at_1(references, predictions):
    from lm_eval.tasks.mbpp import utils

    return utils.pass_at_1(references, predictions)


def list_fewshot_samples():
    from lm_eval.tasks.mbpp import utils

    return utils.list_fewshot_samples()


def extract_code(text: str) -> str:
    """The gen_prefix already opened a ```python block, so the code runs up to the closing fence."""
    text = text.lstrip("\n")
    if text.startswith("```"):
        # The model reopened the block itself; drop its fence line
        text = text.split("\n", 1)[1] if "\n" in text else ""
    return text.split("```")[0].rstrip()


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_code(r) for r in resp] for resp in resps]
