import dspy


lm = dspy.LM("huggingface/Qwen/Qwen2.5-7B-Instruct", api_key="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")
dspy.configure(lm=lm)
# messages = [
#     {
#         "role": "system",
#         "content": "You are a helpful assistant. You speak french"
#     },
#     {
#         "role": "user",
#         "content": "What is the capital of France?"
#     }
# ]

# print(lm(messages = messages))

# haiku_signature = "language, subject -> description, price"
# haiku_generator = dspy.Predict(haiku_signature)
# result = haiku_generator(subject="samsung galaxy s23", language="french")
# print(result.description)
# print("---")
# print(result.price)
# print(dspy.inspect_history())


class DescripionBot(dspy.Signature):
    """Write a precise description in French of the given subject."""
    subject: str = dspy.InputField()
    description:str = dspy.OutputField()


description_bot = dspy.Predict(DescripionBot)
result = description_bot(subject="samsung galaxy s23")
print(result.description)