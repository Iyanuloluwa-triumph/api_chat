import base64
import os
import re
from groq import Groq

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def extract_image_description(image_bytes: bytes) -> str:
  base64_image = base64.b64encode(image_bytes).decode("utf-8")

  try:
    response = groq_client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an image tagging assistant. Skip internal"
                    " reasoning. Do NOT output <think> tags. Respond strictly"
                    " with 3 to 6 space-separated keywords describing the"
                    " main product."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Identify the main product in 3 to 6 search terms"
                            " (e.g., 'electric glass kettle hot water')."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            },
        ],
        temperature=0.1,
        max_tokens=400,  # Fast response time without hitting token caps
    )

    raw_text = response.choices[0].message.content or ""

    # 1. If </think> exists, take everything after it
    if "</think>" in raw_text:
      clean_text = raw_text.split("</think>")[-1].strip()
    else:
      # 2. If truncated mid-thought, grab ONLY the last line where it states candidate terms
      lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
      clean_text = lines[-1] if lines else ""

      # Strip conversational CoT prefixes from the last line
      clean_text = re.sub(
          r"^(terms:|lets try:|let's try:|\d+\.)",
          "",
          clean_text,
          flags=re.IGNORECASE,
      ).strip()

    # 3. Purge all tags, markdown, punctuation, and extra whitespace
    clean_text = re.sub(r"<[^>]+>", "", clean_text)
    clean_text = re.sub(r"[\*\_`#\"'()<>:,]", "", clean_text)
    clean_text = " ".join(clean_text.split())

    print(f"Groq Vision extracted tags: '{clean_text}'")
    return clean_text if clean_text else "kettle water boiler"

  except Exception as e:
    print(f"Groq Vision Error: {type(e).__name__} - {e}")
    return "general store product"