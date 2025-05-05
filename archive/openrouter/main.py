import sys
from openrouter.pipelines.chat_pipeline import chat

def main():
    if len(sys.argv) > 1:
        user_input = sys.argv[1]
    else:
        user_input = input("Enter your prompt: ")
    response = chat(user_input, config_path="configs/openrouter_config.yaml")
    print(f"Response!: {response}")

if __name__ == "__main__":
    main()