# TinyLlama Fine-Tuning Pipeline with QLoRA

## What this project does
Fine-tuned TinyLlama-1.1B on Alpaca dataset using QLoRA (4-bit quantization + LoRA adapters).
Quantized to GGUF format and served locally using Ollama and llama-cpp-python.

## Pipeline
1. Dataset preparation and cleaning from Hugging Face Hub (yahma/alpaca-cleaned)
2. Fine-tuning with QLoRA using PEFT + TRL on Google Colab T4 GPU
3. Quantization to GGUF format using llama.cpp
4. Local serving and evaluation via Ollama and llama-cpp-python

## Tech Stack
Python, HuggingFace Transformers, PEFT, TRL, BitsAndBytes, llama.cpp, Ollama

## Download Model
Download the GGUF model from Hugging Face:
https://huggingface.co/31mohit/tinyllama-finetune

## How to run
1. Download the GGUF model from Hugging Face link above
2. Install Ollama from https://ollama.com
3. Create a Modelfile with: FROM ./tinyllama-finetuned.gguf
4. Run: ollama create my-tinyllama -f Modelfile
5. Run: ollama run my-tinyllama

## Results
- Successfully fine-tuned TinyLlama-1.1B on 1,000 instruction-response pairs
- Quantized model size: 2.2GB (GGUF f16 format)
- Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Dataset: yahma/alpaca-cleaned
```

Press **Ctrl + S**, then run in terminal:
```
git add README.md
git commit -m "Update README with model and run instructions"
git push