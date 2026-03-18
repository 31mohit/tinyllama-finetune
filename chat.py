from llama_cpp import Llama

print("Loading model... please wait...")
llm = Llama(
    model_path="./tinyllama-finetuned.gguf",
    n_gpu_layers=0,
    n_ctx=512
)

print("✅ Model loaded! Type your message below.")
print("Type 'exit' to quit\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    
    response = llm(
        f"### Instruction:\n{user_input}\n\n### Response:\n",
        max_tokens=200,
        stop=["### Instruction:"],
        echo=False
    )
    
    print(f"Model: {response['choices'][0]['text']}\n")

