from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B")
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt},
]
pipe(messages)
