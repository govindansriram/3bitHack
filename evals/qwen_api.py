"""lm-eval model that talks to a vLLM OpenAI-compatible server through the /v1/completions endpoint.

The chat template is rendered on the client with the Qwen3 tokenizer, so we control thinking mode
directly (the server only ever sees token ids). Using /completions instead of /chat/completions also
keeps batching and prompt logprobs available, so one model class serves both loglikelihood tasks
(wikitext, mmlu, arc) and generation tasks (gsm8k, humaneval, mbpp).

Registered as "qwen3-vllm". Extra model_args on top of local-completions:
  enable_thinking        render prompts with Qwen3 thinking on (default False)
  thinking_max_gen_toks  generation budget when thinking is on (default 16384)
"""
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalCompletionsAPI

THINK_END = "</think>"
# vLLM's OpenAI API rejects requests with more than 4 stop strings (HumanEval has 5)
MAX_SERVER_STOPS = 4


@register_model("qwen3-vllm")
class Qwen3VLLM(LocalCompletionsAPI):
    def __init__(self, enable_thinking=False, thinking_max_gen_toks=16384, **kwargs):
        super().__init__(**kwargs)
        self.enable_thinking = str(enable_thinking).lower() in ("1", "true", "yes")
        self.thinking_max_gen_toks = int(thinking_max_gen_toks)
        self._stops = []

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        # When a task prefills the assistant turn (gen_prefix), Qwen3's template always inserts an
        # empty <think></think> block before the prefix, so those tasks run without thinking.
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            enable_thinking=self.enable_thinking,
        )

    def _create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None, **kwargs):
        if generate and self.enable_thinking:
            # Task stop strings ("\n\n", "Question:") would fire inside the reasoning trace, so stop
            # only at end of turn and give the model room to think.
            gen_kwargs = dict(gen_kwargs or {})
            gen_kwargs["until"] = []
            gen_kwargs["max_gen_toks"] = max(gen_kwargs.get("max_gen_toks", 0), self.thinking_max_gen_toks)
        payload = super()._create_payload(messages, generate=generate, gen_kwargs=gen_kwargs, seed=seed, eos=eos,
                                          **kwargs)
        if generate:
            # The server applies the first few stops; parse_generations applies the full list. Every request
            # in a task shares the same generation kwargs, so remembering the latest list is enough.
            stops = payload.get("stop") or []
            self._stops = [stops] if isinstance(stops, str) else list(stops)
            payload["stop"] = self._stops[:MAX_SERVER_STOPS]
        return payload

    def parse_generations(self, outputs, **kwargs):
        texts = LocalCompletionsAPI.parse_generations(outputs, **kwargs)
        if self.enable_thinking:
            # Score only the final answer, not the reasoning trace. Leave replies without a trace untouched:
            # stripping them would also strip leading code indentation.
            texts = [t.split(THINK_END)[-1].lstrip() if t is not None and THINK_END in t else t for t in texts]
        return [self._apply_stops(t) for t in texts]

    def _apply_stops(self, text):
        if text is None:
            return text
        for stop in self._stops:
            if stop:
                text = text.split(stop)[0]
        return text
