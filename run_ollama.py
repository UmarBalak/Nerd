import re
import chromadb
from langchain_ollama import OllamaLLM

class OllamaAssistant:
    def __init__(self, model="llama3.2:latest", system_prompt=None, db_path="./chroma_db"):
        self.model = model

        # Load system prompt from file
        self.system_prompt = system_prompt or self.load_system_prompt()

        # Initialize chromadb client
        self.client = chromadb.PersistentClient(db_path)
        self.collection = self.client.get_or_create_collection("conversations_history")
    
    def load_system_prompt(self):
        try:
            with open("system_prompt.txt", "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            return ""

    def store_conversation(self, user_query, assistant_response):
        """Stores the conversation in ChromaDB."""
        self.collection.add(
            documents=[user_query, assistant_response], # Text data
            ids=[f"user-{len(self.collection.get()['ids'])}",
                 f"assistant-{len(self.collection.get()['ids'])}"] # Unique IDs for user and assistant
        )
    
    def get_recent_context(self, num_messages=20):
        """Retrieves the last 'num_messages' from the conversation history."""
        all_data = self.collection.get()
        messages = list(zip(all_data["ids"], all_data["documents"]))
        return "\n".join([msg[1] for msg in messages[-num_messages:]])

    def get_response(self, query):
        # Retrive recent history from chromadb
        recent_context = self.get_recent_context()

        # Format the full context
        full_context = f"{self.system_prompt}\n\nRecent Conversation History:\n{recent_context}\n\nUser Query:\n{query}"
        
        llm = OllamaLLM(model=self.model)
        response = llm.invoke(full_context)
        
        # Store the new conversation
        self.store_conversation(query, response)
        
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