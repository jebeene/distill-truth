import argparse
from archive.local.loader import load_generation_model
from config import DEFAULT_MODEL_NAME

def chat(model_name):
    tokenizer, model, device = load_generation_model(model_name)
    print(f"\nChatting with {model_name}. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break

        prompt = f"{user_input.strip()}"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"🤖: {response[len(prompt):].strip()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="HuggingFace model ID")
    args = parser.parse_args()

    chat(args.model)
