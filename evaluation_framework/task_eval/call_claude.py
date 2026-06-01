import os
import time
import anthropic

MAX_RETRIES = 5
BASE_DELAY = 3.0


def call_claude(input_prompt: str, model: str, **kwargs) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY must be set.")

    client = anthropic.Anthropic(api_key=api_key)

    from task_eval.utils import _build_model_input, _prepend_conv_prefix

    category = kwargs.get("category", "")
    content = _build_model_input(input_prompt or "", category=category) if category else _prepend_conv_prefix(input_prompt or "")

    temperature = kwargs.get("temperature", 0.3)
    max_tokens = kwargs.get("max_tokens", 2048)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                messages=[{"role": "user", "content": content}],
            )
            text = (response.content[0].text or "").strip()
            return text if text else "(empty)"
        except anthropic.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
                continue
            return "[API Error: Rate limit exceeded after retries]"
        except Exception as e:
            return f"[API Error: {e}]"

    return "[API Error: Max retries reached]"


def generate_responses(prompts: list, model: str, **kwargs) -> list:
    responses = []
    for prompt in prompts:
        res = call_claude(prompt, model, **kwargs)
        responses.append(res)
    return responses
