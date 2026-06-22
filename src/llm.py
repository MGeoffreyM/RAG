from transformers import AutoModelForCausalLM, AutoTokenizer


class Llm():
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def prompting(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # thinking_content = self.tokenizer.decode(output_ids[:index],
        #                                          skip_special_tokens=True)
        content = self.tokenizer.decode(output_ids[index:],
                                        skip_special_tokens=True)
        return content
