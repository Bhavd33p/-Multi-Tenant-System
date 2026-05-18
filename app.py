import time
import asyncio
import torch
from threading import Thread
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import LoraConfig, PeftModel

app = FastAPI(
    title="Multi-Adapter Engine (Apple Silicon Optimized)",
    description="Serving multiple specialized adapters on a single Mac GPU pool."
)

# 1. Verify Mac GPU Acceleration (MPS)
if not torch.backends.mps.is_available():
    raise RuntimeError("MPS backend is not available. Ensure you are on an Apple Silicon Mac.")
device = torch.device("mps")
print(f"🍏 Apple Silicon Detected. Directing tensor operations to: {device}")

# 2. Load the Shared Brain (Base Model) into Unified Memory
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"📥 Downloading/Loading Base Brain: {BASE_MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# float16 optimization uses half the RAM, leaving plenty of room on your Mac
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)

# 3. Create the Structural Adapter Slots
# We freeze the base model and hook up two adapter configs
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"
)

print("🛠️ Injecting adapter slots into memory layout...")
model = PeftModel(base_model, lora_config, adapter_name="adapter_code")
model.add_adapter("adapter_finance", lora_config)

# 4. The Concurrency Guard (The Traffic Cop)
# Prevents two concurrent requests from swapping adapters mid-generation
model_lock = asyncio.Lock()

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7

async def stream_inference(adapter_name: str, prompt: str, max_tokens: int, temperature: float):
    # Form a neat line; lock the model state until this generation finishes
    async with model_lock:
        start_time = time.time()
        
        # Instantly swap pointers to the target adapter
        model.set_adapter(adapter_name)
        model.eval()
        
        # Apply standard system instruction templates
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Set up an async word-by-word streamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Run generation on a background execution thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        token_count = 0
        yield f"[METRIC] Active Adapter: {adapter_name} | Swapped instantly via Unified Memory\n\n"
        
        for new_text in streamer:
            token_count += 1
            yield new_text
            await asyncio.sleep(0.001) # Yield priority to the API event loop

@app.post("/v1/generate")
async def generate_endpoint(request: GenerationRequest, x_adapter_target: str = Header(...)):
    target_adapter = x_adapter_target.strip().lower()
    
    if target_adapter not in ["adapter_code", "adapter_finance"]:
        raise HTTPException(
            status_code=400, 
            detail="Header 'X-Adapter-Target' must be either 'adapter_code' or 'adapter_finance'."
        )
        
    return StreamingResponse(
        stream_inference(
            adapter_name=target_adapter,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        ),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, workers=1)