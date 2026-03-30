import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, GenerationConfig

# USE V0.3 OR LATEST, BUT ENSURE AUTHENTICATION
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" 

def run_rpg_system():
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return
    
    print(f"--- Loading Mug the Orc ({torch.cuda.get_device_name(0)}) ---")
    
    # 1. LOAD MODEL AND TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # Use eager if sdpa causes errors on older GPUs
        attn_implementation="sdpa" 
    )
    
    # 2. CREATE A CLEAN CONFIG
    gen_config = GenerationConfig(
        max_new_tokens=60, # Increased for better answers
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # System prompt
    history = "System: You are Mug, a rude Orc bartender. You hate humans. Speak ONLY as Mug. One or two gruff sentences. Never repeat yourself.\n"
    
    print("\n--- The Broken Tusk Tavern ---")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            break
            
        history += f"Traveler: {user_input}\n"
        
        # Correctly formatted Mistral Prompt
        prompt = f"<s>[INST] {history} [/INST]\nMug: "
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Clear terminal
        os.system('cls' if os.name == 'nt' else 'clear')
        print("--- The Broken Tusk Tavern ---")
        print(f"You: {user_input}")
        print("Mug: ", end="", flush=True)
        
        # 3. GENERATE
        _ = model.generate(
            **inputs, 
            generation_config=gen_config, 
            streamer=streamer
        )
        # Update history with the last turn for context
        # (Simplified, normally you'd extract the new response)
        history += "Mug: ...\n"

if __name__ == "__main__":
    run_rpg_system()
