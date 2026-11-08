
from openai import OpenAI
import json
 
from config import (
    # NVIDIA_API_KEY,
    HOOSHYAR_API_KEY,
    BASE_URL,
    MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    REASONING
)


client = OpenAI(
    # api_key=NVIDIA_API_KEY,
    api_key=HOOSHYAR_API_KEY,
    base_url=BASE_URL
)


def ask_llm(
    prompt,
    seed=42
):

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=TEMPERATURE,

        top_p=0.95,

        max_tokens=MAX_TOKENS,

        seed=seed,

        stream=False,

        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": REASONING
            }
        }
    )

    # return response.choices[0].message.content.strip()

    content = extract_final_text(response)

    if content is None or not content.strip():
        raise ValueError("LLM returned empty content")

    return content.strip()


def extract_final_text(raw_response):
    parts = []

    for line in raw_response.splitlines():
        line = line.strip()

        if not line.startswith("data:"):
            continue

        data = line[len("data:"):].strip()

        if data == "[DONE]":
            break

        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        for choice in event.get("choices", []):
            content = choice.get("delta", {}).get("content")

            if isinstance(content, str):
                parts.append(content)

    return "".join(parts)   

