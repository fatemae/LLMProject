from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from colorama import Fore

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")
# prompt_template = PromptTemplate(
#     input_variables=["topic"],
#     template="Tell me a short joke about {topic}",
# )
prompt_template = ChatPromptTemplate.from_template("Tell me a short joke about {topic}?")
output_parser = StrOutputParser()

def generate(text):
    """ generate text based on the input """
    # prompt = prompt_template.format(topic=text)
    # return llm.invoke(prompt).content

    chain = prompt_template | llm | output_parser
    response = chain.invoke({"topic": text})
    return response.content


def start():
    instructions = (
        "Type your question and press ENTER. Type 'x' to go back to the MAIN menu.\n"
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

            response = generate(user_input)
            print(Fore.BLUE + f"A: " + response + Fore.RESET)
            print(Fore.WHITE + "\n-------------------------------------------------")


if __name__ == "__main__":
    start()
