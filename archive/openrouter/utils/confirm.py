def ask_approval(prompt="Continue? (y/n): "):
    while True:
        choice = input(prompt).strip().lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            print("Please respond with 'y' or 'n'.")