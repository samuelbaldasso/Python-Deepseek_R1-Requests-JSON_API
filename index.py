import requests
import json
import os
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Message:
    role: str
    content: str

class DeepseekClient:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
            
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def chat(self, messages: List[Message], model: str = "deepseek-chat", temperature: float = 0.7) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": model,
                    "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
                    "temperature": temperature
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error decoding response: {e}")
            return None

    def save_response(self, response: Dict[str, Any], filename: str = "response.json") -> None:
        try:
            with open(filename, "w") as f:
                json.dump(response, f, indent=2)
            print(f"Response saved to {filename}")
        except IOError as e:
            print(f"Error saving response: {e}")

def main():
    try:
        # Initialize client
        client = DeepseekClient()

        # Create message
        messages = [Message(role="user", content="Hello, how are you?")]

        # Make request
        response = client.chat(messages)
        
        if response:
            print("API Response:")
            print(json.dumps(response, indent=2))
            
            # Save response
            client.save_response(response)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()