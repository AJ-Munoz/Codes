import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, GenerationConfig

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

def run_rpg_system():
    if not torch.cuda.is_available(): return

    print(f"--- Loading Mug the Orc ({torch.cuda.get_device_name(0)}) ---")
    
    # 1. LOAD MANUALLY (Cleaner than pipeline for high-end GPUs)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    # 2. CREATE A CLEAN CONFIG (This stops all the warnings)
    gen_config = GenerationConfig(
        max_new_tokens=45,
        do_sample=True,
        temperature=0.8,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    history = "System: You are Mug, a rude Orc bartender. You hate humans. Speak ONLY as Mug. One or two gruff sentences. Never repeat yourself.\n"

    #os.system('cls' if os.name == 'nt' else 'clear')
    print("\n--- The Broken Tusk Tavern ---")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]: break

        history += f"Traveler: {user_input}\n"
        prompt = f"[INST] {history} [/INST]\nMug:"

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"--- The Broken Tusk Tavern ---")
        print(f"You: {user_input}")
        print(f"Mug: ", end="", flush=True)

        # 3. GENERATE (Using the manual model call)
        output_tokens = model.generate(
            **inputs,
            generation_config=gen_config,
            streamer=streamer
        )

        # Decode only the NEW tokens
        new_tokens = output_tokens[0][inputs['input_ids'].shape[-1]:]
        new_res = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Safety split
        new_res = new_res.split("Traveler:")[0].split("You:")[0].strip()
        
        history += f"Mug: {new_res}\n"
        if len(history) > 1000: history = history[-1000:] 

if __name__ == "__main__":
    run_rpg_system()
