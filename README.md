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

## How to run
1. Install Ollama from https://ollama.com
2. Run: ollama create my-tinyllama -f Modelfile
3. Run: ollama run my-tinyllama