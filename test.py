import packaging.version

# --- CORRECTIF POUR LE BUG TRANSFORMERS / GGUF ---
# Contournement du crash "InvalidVersion: 'N/A'"
_original_parse = packaging.version.parse
def _safe_parse(v):
    if v == 'N/A':
        return _original_parse('99.9.9')
    return _original_parse(v)
packaging.version.parse = _safe_parse
# -------------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Définir les noms
base_model_name = "Qwen/Qwen3-0.6B" # Pour le tokenizer (dictionnaire)
gguf_repo_name = "Qwen/Qwen3-0.6B-GGUF" # Le dépôt contenant les GGUF
# ⚠️ Remplace par le VRAI nom du fichier sur le site Hugging Face (onglet "Files")
gguf_filename = "Qwen3-0.6B-Q8_0.gguf" 

print("Chargement du dictionnaire (tokenizer)...")
# On charge le tokenizer depuis le modèle de base (souvent absent des repos GGUF)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print("Chargement du cerveau (modèle GGUF)...")
# On charge le modèle depuis le repo GGUF en spécifiant le fichier
model = AutoModelForCausalLM.from_pretrained(
    gguf_repo_name,
    gguf_file=gguf_filename,  # <-- La ligne magique pour le GGUF
    torch_dtype="auto",
    device_map="auto",
    offload_folder="offload"
)

# ... LE RESTE DE TON CODE RESTE STRICTEMENT IDENTIQUE ...

# 2. Préparer le message (Prompt)
messages = [
    {"role": "user", "content": "Explique-moi comment fonctionne un vélo."}
]

# On applique le format de dialogue spécifique au modèle
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 3. La Génération
print("Génération de la réponse...")
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048 # (J'ai baissé un peu pour tes tests, 32k c'est énorme et lent !)
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# 4. Parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
