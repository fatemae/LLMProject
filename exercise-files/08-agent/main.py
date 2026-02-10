from colorama import Fore
from agent import run_agent
import time

start_time = time.time()


def parse_stream_response(response):
    """Iterate through the stream of events"""
    collected_chunks = []
    collected_messages = []

    for chunk in response:
        chunk_time = time.time() - start_time  # calculate the time delay of the chunk
        collected_chunks.append(chunk)  # save the event response
        print(chunk)
        output = chunk["output"]
        collected_messages.append(output)  # save the message
        
    # print the time delay and text received
    print(f"Full response received {chunk_time:.2f} seconds after request")
    # clean None in collected_messages
    collected_messages = [m for m in collected_messages if m is not None]
    full_reply_content = ''.join(collected_messages)
    print(f"Full conversation received: {full_reply_content}")
    return full_reply_content

def start():
    instructions = (
        """Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"""
    )
    print(Fore.BLUE + "\n\x1B[3m" + instructions + "\x1B[0m" + Fore.RESET)

    print("MENU")
    print("====")
    print("[1]- Ask a question")
    print("[2]- Exit")
    choice = input("Enter your choice: ")
    if choice == "1":
        ask()
    elif choice == "2":
        print("Goodbye!")
        exit()
    else:
        print("Invalid choice")
        start()


def ask():
    while True:
        user_input = input("Q: ")
        # Exit
        if user_input == "x":
            start()
        else:

            response = run_agent(user_input)
            print(Fore.BLUE + "A: " + response + Fore.RESET)
            print(Fore.WHITE + 
                  "\n-------------------------------------------------")


if __name__ == "__main__":
    start()