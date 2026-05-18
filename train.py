import os
import torch
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

print("🚀 Initializing Day 3 Native MPS Fine-Tuning Engine...")

# 1. Device Targeting for Apple Silicon
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS backend not found. Ensure you are on an Apple Silicon Mac.")
device = torch.device("mps")

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# 2. Simple Data Collator to handle matrix dimensions
def custom_data_collator(features):
    """
    Takes a batch of tokenized samples and pads them to the same length
    so they form a perfect uniform matrix for the Mac GPU.
    """
    batch = {}
    # Extract the maximum sequence length in this specific batch
    max_len = max(len(feature["input_ids"]) for feature in features)
    
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    
    for feature in features:
        pad_len = max_len - len(feature["input_ids"])
        
        # Pad with 0s for inputs and masks (assuming 0 is safe for padding)
        padded_inputs = torch.cat([feature["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        padded_mask = torch.cat([feature["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        
        # In causal LM training, labels are an exact copy of input_ids. 
        # We fill padding zones with -100 so the PyTorch Loss function ignores them.
        padded_labels = torch.cat([feature["input_ids"], torch.full((pad_len,), -100, dtype=torch.long)])
        
        batch_input_ids.append(padded_inputs)
        batch_attention_mask.append(padded_mask)
        batch_labels.append(padded_labels)
        
    batch["input_ids"] = torch.stack(batch_input_ids).to(device)
    batch["attention_mask"] = torch.stack(batch_attention_mask).to(device)
    batch["labels"] = torch.stack(batch_labels).to(device)
    return batch

def run_adapter_training(tokenized_data_path, output_adapter_dir):
    """
    Loads pre-computed matrix binaries and runs an isolated gradient optimization
    loop to train a lightweight parameter adapter module.
    """
    print(f"\n📂 Loading Dataset: {tokenized_data_path}")
    dataset = torch.load(tokenized_data_path, weights_only=False)
    
    print(f"📥 Loading Base Brain weights onto MPS...")
    # Load base brain weights in float16 to conserve memory space
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    
    # 3. Define the LoRA Parameter boundaries
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"], # Inject exclusively into attention gates
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    
    # Wrap the base model with our trainable LoRA framework configuration
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters() # Prints exactly how tiny the adapter is!
    
    # 4. Configure Training Hyperparameters optimized for Mac hardware
    training_args = TrainingArguments(
        output_dir=output_adapter_dir,
        per_device_train_batch_size=1,        # Small batch size to strictly prevent VRAM crashes
        gradient_accumulation_steps=2,       # Simulates a batch size of 2 safely
        num_train_epochs=3,                  # Loop over our dataset 3 times
        learning_rate=2e-4,                  # Standard stable step size for adapters
        fp16=False,                          # Keep False: Native half-precision training on MPS utilizes custom torch configurations
        logging_steps=1,
        save_strategy="no",                  # Only save at the very end to prevent disk lag
        report_to="none",
        use_cpu=False                        # Force calculation execution explicitly on the Mac GPU
    )
    
    # 5. Initialize Optimization Loop Controller
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=custom_data_collator,
    )
    
    print("🏋️ Training initialized. Executing gradient steps on Mac GPU cores...")
    trainer.train()
    
    # 6. Save only the lightweight adapter weights to disk, keeping base model clean
    model.save_pretrained(output_adapter_dir)
    print(f"💾 Adapter successfully trained and saved -> {output_adapter_dir}")

if __name__ == "__main__":
    # Run the optimization loop completely independently for both streams
    print("--- 🛠️ STEP 1: Training Code Specialist Adapter ---")
    run_adapter_training("finetuning/tokenized_code.pt", "finetuning/adapter_code")
    
    print("\n--- 🛠️ STEP 2: Training Finance Specialist Adapter ---")
    run_adapter_training("finetuning/tokenized_finance.pt", "finetuning/adapter_finance")