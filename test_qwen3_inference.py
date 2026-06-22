import argparse
import time
import sys

def test_vllm(model_name, prompt):
    print(f"\n{'='*50}")
    print(f"🚀 Lancement de l'inférence avec vLLM pour {model_name}")
    print(f"{'='*50}\n")
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Erreur : vllm n'est pas installé. Lancez `pip install vllm`")
        sys.exit(1)
    
    start_load = time.time()
    # Chargement du modèle
    llm = LLM(model=model_name, trust_remote_code=True)
    print(f"✅ Modèle chargé en {time.time() - start_load:.2f} secondes.")
    
    # Paramètres de génération
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
    
    start_gen = time.time()
    # Inférence
    outputs = llm.generate([prompt], sampling_params)
    gen_time = time.time() - start_gen
    
    generated_text = outputs[0].outputs[0].text
    
    print(f"\n⏱️ Temps de génération : {gen_time:.2f}s")
    print(f"💬 Réponse générée :\n{generated_text}\n")


def test_sglang(model_name, prompt):
    print(f"\n{'='*50}")
    print(f"🚀 Lancement de l'inférence avec SGLang pour {model_name}")
    print(f"{'='*50}\n")
    try:
        import sglang as sgl
    except ImportError:
        print("Erreur : sglang n'est pas installé. Lancez `pip install sglang`")
        sys.exit(1)
    
    start_load = time.time()
    # Initialisation du moteur hors ligne
    engine = sgl.Engine(model_path=model_name)
    print(f"✅ Modèle chargé en {time.time() - start_load:.2f} secondes.")
    
    # Paramètres de génération (SGLang utilise max_new_tokens)
    sampling_params = {"temperature": 0.7, "max_new_tokens": 256}
    
    start_gen = time.time()
    # Inférence
    response = engine.generate(
        prompt=prompt,
        sampling_params=sampling_params
    )
    gen_time = time.time() - start_gen
    
    print(f"\n⏱️ Temps de génération : {gen_time:.2f}s")
    print(f"💬 Réponse générée :\n{response['text']}\n")
    
    # Il est recommandé de fermer le moteur proprement avec SGLang
    engine.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de test d'inférence pour Qwen3-0.6B")
    parser.add_argument("--backend", type=str, choices=["vllm", "sglang"], required=True, 
                        help="Le moteur d'inférence à tester (vllm ou sglang)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", 
                        help="Nom ou chemin d'accès au modèle sur Hugging Face")
    parser.add_argument("--prompt", type=str, default="Donne-moi une courte introduction sur les modèles de langage (LLM).", 
                        help="Le prompt à envoyer au modèle")
    
    args = parser.parse_args()
    
    if args.backend == "vllm":
        test_vllm(args.model, args.prompt)
    elif args.backend == "sglang":
        test_sglang(args.model, args.prompt)
