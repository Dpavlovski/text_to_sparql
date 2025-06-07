from src.agent.workflow import app

if __name__ == '__main__':
    question = "Who is the director of the movie Inception?"
    inputs = {"question": question}

    for event in app.stream(inputs, {"recursion_limit": 10}):
        for key, value in event.items():
            print(f"--- Event: {key} ---")
            print(value)
            print("\n")
