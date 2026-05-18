import json
import torch
from transformers import AutoTokenizer

print("🔄 Initializing Day 2 Data Preprocessing Engine...")

# Load the matching tokenizer configuration for our shared base model brain
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def process_and_tokenize_file(input_json_path, output_pt_path):
    """
    Reads a conversation JSON, applies the structural chat template syntax,
    and formats the data into tensor arrays ready for local GPU training loops.
    """
    with open(input_json_path, "r") as f:
        raw_data = json.load(f)
        
    tokenized_samples = []
    
    for item in raw_data:
        # 1. Inject the text into the official ChatML assistant wrapper template
        formatted_text = tokenizer.apply_chat_template(
            item["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # 2. Transform strings into input integer token IDs and attention masks
        tokenized_output = tokenizer(
            formatted_text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Squeeze out the extra batch dimensions added by default tokenizer settings
        sample_dict = {
            "input_ids": tokenized_output["input_ids"].squeeze(0),
            "attention_mask": tokenized_output["attention_mask"].squeeze(0)
        }
        tokenized_samples.append(sample_dict)
        
    # Serialize the tensor structural matrix arrays directly onto your Mac disk
    torch.save(tokenized_samples, output_pt_path)
    print(f"✅ Preprocessing Complete: {len(raw_data)} samples saved -> {output_pt_path}")

if __name__ == "__main__":
    # Execute preprocessing pipelines completely independently 
    process_and_tokenize_file("finetuning/dataset_code.json", "finetuning/tokenized_code.pt")
    process_and_tokenize_file("finetuning/dataset_finance.json", "finetuning/tokenized_finance.pt")