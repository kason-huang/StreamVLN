from transformers import AutoModel, AutoTokenizer
import os

model_id = "google/siglip-so400m-patch14-384"

try:
    print(f"Testing HF model download: {model_id}")
    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("✅ Successfully downloaded model and tokenizer from Hugging Face Hub.")
except Exception as e:
    print("❌ Failed to download model.")
    print("Error:", str(e))
