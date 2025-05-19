import requests
from typing import Any, Optional, Tuple

from chat_checker.chatbot_connection.chatbot_client_base import ChatbotClientInterface
from chat_checker.chatbot_connection.chatbot_client_woz_test import chat_with_chatbot


class ChatbotClient(ChatbotClientInterface):
    BASE_URL = "http://127.0.0.1:5000/api/v2"
    IS_STUPID = False
    CD_EXAMPLES = True
    FEEDBACK = True
    GOODBYE_MESSAGE = (
        "Goodbye! I hope I was able to help you with your goal. Have a great day!"
    )

    def __init__(self):
        self.session_id = None

    def set_up_chat(self) -> Optional[str]:
        print("Setting up chat session...")
        initialization_payload = {
            "is_stupid": self.IS_STUPID,
            "cd_examples": self.CD_EXAMPLES,
            "feedback": self.FEEDBACK,
        }
        response = requests.post(
            f"{self.BASE_URL}/create-assistant", json=initialization_payload
        )
        response.raise_for_status()
        init_response = response.json()
        self.session_id = init_response.get("id")
        first_chatbot_message = init_response["message_history"][0]["content"]
        print(f"Chat setup complete with session ID: {self.session_id}")
        return first_chatbot_message

    def tear_down_chat(self, *args) -> Any:
        print("Tearing down chat session...")
        # Delete the created chat session
        deletion_payload = {"session_id": self.session_id}
        response = requests.post(
            f"{self.BASE_URL}/delete-chat-session", json=deletion_payload
        )
        response.raise_for_status()
        chatbot_response = response.json()
        self.session_id = None
        print(f"Chat teardown complete: {chatbot_response}")
        return chatbot_response

    def get_response(self, user_message: str) -> Tuple[str, bool]:
        if not self.session_id:
            raise ValueError(
                "Chat session has not been initialized. Please set up the chat first."
            )

        chat_payload = {"session_id": self.session_id, "message": user_message}
        response = requests.post(f"{self.BASE_URL}/get-answer", json=chat_payload)
        response.raise_for_status()
        chatbot_response: dict = response.json()
        chatbot_message = chatbot_response.get("message_history", [{}])[-1].get(
            "content", ""
        )
        is_finished = chatbot_message == self.GOODBYE_MESSAGE
        return chatbot_message, is_finished


# Example usage
if __name__ == "__main__":
    chatbot_client = ChatbotClient()

    chat_with_chatbot(chatbot_client)
