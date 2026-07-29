import torch
from PIL import Image
from typing import Any
from transformers import (AutoProcessor,
                          AutoModelForImageTextToText,
                          AutoTokenizer,
                          AutoModelForCausalLM)


class AnswerLlm():
    def __init__(self, model_name: str) -> None:
        """
        Initialise le modèle de langage (LLM) et son tokenizer.

        Cette méthode instancie le tokenizer via 'AutoTokenizer' et le modèle
        de génération de texte via 'AutoModelForCausalLM'.
        L'allocation du modèle sur les dispositifs matériels (CPU/GPU) se fait
        automatiquement via 'device_map="auto"'.

        Args:
            model_name (str): L'identifiant ou le chemin du modèle à charger.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def prompting(self,
                  prompt: str) -> tuple[str | list[str], str | list[str]]:
        """
        Génère une réponse à partir d'un prompt utilisateur en séparant le
        contenu final et le raisonnement (thinking).

        La méthode formate la requête selon le template de chat du modèle.
        Une fois les tokens générés (limités à 2048 nouveaux tokens),
        elle identifie la séparation entre la phase de réflexion et la réponse
        finale grâce à la présence du token '151668'.
        Les deux segments sont ensuite décodés en ignorant les tokens spéciaux.

        Args:
            prompt (str): La requête utilisateur soumise au modèle[cite: 4].

        Returns:
            tuple[str | list[str], str | list[str]]: Un tuple dont le premier
                élément est le contenu final de la réponse et le second le
                contenu de réflexion du modèle.
        """
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
        """
        Initialise le modèle de vision et de langage, le processeur et le
        tokenizer par défaut.

        Le modèle utilisé par défaut est 'HuggingFaceTB/SmolVLM-256M-Instruct'.
        Il est chargé avec 'AutoModelForImageTextToText' au format de
        précision 'bfloat16', accompagné de son processeur ('AutoProcessor')
        et de son tokenizer ('AutoTokenizer').
        """
        self.model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def describe_image(self, image_path: str) -> str:
        """
        Analyse une image spécifiée par son chemin d'accès et produit une
        description détaillée, concise et technique.

        L'image est d'abord chargée, convertie au format RGB, puis
        redimensionnée en miniature (maximum 512x512 pixels)
        avant d'être soumise au modèle.
        La requête est traitée via le processeur pour inclure l'image
        et l'instruction textuelle dans le template de chat.

        Args:
            image_path (str): Le chemin local de l'image à décrire.

        Returns:
            str: Le texte décrivant l'image généré par le modèle,
            ou une chaîne de caractères formatant l'exception en cas
            d'erreur lors du traitement.
        """
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

            output_text: str | list[str] = self.tokenizer.decode(
                generated_ids_trimmed[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            return str(output_text)

        except Exception as e:
            return f"Erreur lors de l'analyse de l'image : {e}"
