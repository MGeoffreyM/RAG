import torch
from PIL import Image
from transformers import (AutoProcessor,
                          AutoModelForImageTextToText,
                          AutoTokenizer,
                          AutoModelForCausalLM)


class AnswerLlm():
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def prompting(self,
                  prompt: str) -> tuple[str | list[str], str | list[str]]:
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
            max_new_tokens=2048
        )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index],
                                                 skip_special_tokens=True)
        content = self.tokenizer.decode(output_ids[index:],
                                        skip_special_tokens=True)
        return (content, thinking_content)


class VisionLlm:
    def __init__(self) -> None:
        self.model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def describe_image(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((512, 512))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image in "
                         "detail. Be concise and technical."}
                    ]
                }
            ]

            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=prompt,
                images=[image],
                return_tensors="pt"
            ).to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=256)

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(
                    inputs.input_ids, generated_ids)
            ]

            output_text: str = self.tokenizer.decode(
                generated_ids_trimmed[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return output_text

        except Exception as e:
            return f"Erreur lors de l'analyse de l'image : {e}"
