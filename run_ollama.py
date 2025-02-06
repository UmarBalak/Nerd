import re
from langchain_ollama import OllamaLLM

class OllamaAssistant:
    def __init__(self, model="llama3.2:latest", system_prompt=None, history_file="conversation_history.txt"):
        self.model = model
        self.conversation_history = []
        self.history_file = history_file
        with open("system_prompt.txt", "r") as file:
            self.system_prompt = file.read()
        if system_prompt:
            self.system_prompt += system_prompt

    def get_response(self, query):
        # Incorporate conversation history for context
        full_context = "\n".join(self.conversation_history + [f"User Query: {query}"])
        
        llm = OllamaLLM(model=self.model)
        response = llm.invoke(f"{self.system_prompt}\n\nConversation History and Current Query:\n{full_context}")
        
        # Store conversation history
        self.conversation_history.append(f"User: {query}")
        self.conversation_history.append(f"Assistant: {response}")
        
        # Write conversation history to file
        with open(self.history_file, "a") as file:
            file.write(f"User: {query}\n")
            file.write(f"Assistant: {response}\n\n")
        
        return response

    def extract_code_snippet(self, response):
        code_matches = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        return code_matches[0] if code_matches else None

# Usage example
assistant = OllamaAssistant()
while True:
    query = input("You: ")
    exit_commands = ["exit", "quit", "goodbye", "bye", "see you"]
    if query.lower() in exit_commands:
        response = assistant.get_response(f"User is exiting the conversation with command: {query}. give a farewell message.")
        print("Ollama:", response)
        break
    response = assistant.get_response(query)
    code_snippet = assistant.extract_code_snippet(response)
    print("Ollama:", response)
    if code_snippet:
        print("Code Snippet:", code_snippet)
    print()