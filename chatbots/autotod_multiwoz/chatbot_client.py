import requests
from typing import Any, Optional, Tuple

from chat_checker.chatbot_connection.chatbot_client_base import ChatbotClientInterface
from chat_checker.chatbot_connection.chatbot_client_woz_test import chat_with_chatbot


class ChatbotClient(ChatbotClientInterface):
    def __init__(self):
        self.base_url = "http://127.0.0.1:8401"
        # self.model = "gpt-3.5-turbo-0613"     # used in AutoTOD-paper (but deprecated)
        # self.model = "gpt-3.5-turbo-1106"     # closest to gpt-3.5 model in paper --> use for gpt-3.5-turbo runs
        # self.model = "gpt-4-0613"             # used in AutoTOD-paper (expensive though)
        self.model = "gpt-4-turbo-2024-04-09"  # used for our paper as more affordable alternative to gpt-4o
        # self.model = "gpt-4o-2024-08-06"      # tried to use for theses but consistently results in chatbot crash due to incorrect output format

    def set_up_chat(self) -> Optional[str]:
        self.session = requests.Session()
        print("Setting up chat session...")
        setup_payload = {"model_name": self.model}
        response = self.session.post(
            f"{self.base_url}/init-session", json=setup_payload
        )
        response.raise_for_status()
        return None

    def tear_down_chat(self, *args) -> Any:
        # Nothing to do here
        pass

    def get_response(self, user_message: str) -> Tuple[str, bool]:
        response = self.session.post(
            f"{self.base_url}/get-answer", json={"user_message": user_message}
        )
        response.raise_for_status()
        chatbot_response: dict = response.json()
        return chatbot_response.get("chatbot_answer", ""), chatbot_response.get(
            "is_finished", False
        )


# Example usage
if __name__ == "__main__":
    chatbot_client = ChatbotClient()

    chat_with_chatbot(chatbot_client)
