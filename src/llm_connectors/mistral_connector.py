from openai import OpenAI
import os

#Llama

from src.llm_connectors.abstract_connector import AbstractConnector


class MistralConnector(AbstractConnector):
    """
    OpenRouter (OpenAI-compatible) connector used in place of the Mistral SDK.
    This allows using Llama (and other models) via OpenRouter while keeping
    FAIRGAME's "MistralConnector" interface unchanged.
    """

    def __init__(self, provider_model: str, temperature: float = 1.0):
        # Keep FAIRGAME's existing env variable name to avoid changing config logic
        self.api_key = os.getenv("API_KEY_MISTRAL")
        if not self.api_key:
            raise EnvironmentError("API_KEY_MISTRAL not found in environment variables.")

        # Default to a free Llama model if not provided by config
        #self.provider_model = provider_model or "meta-llama/llama-3.3-70b-instruct:free"
        self.provider_model = provider_model or "openrouter/free"
        self.temperature = temperature

        # OpenRouter uses an OpenAI-compatible API
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "FAIRGAME_MAS_Collusion"
            }
        )

    def send_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]

        completion = self.client.chat.completions.create(
            model=self.provider_model,
            temperature=self.temperature,
            messages=messages
        )
        return completion.choices[0].message.content



# from mistralai import Mistral
# import os 

# from src.llm_connectors.abstract_connector import AbstractConnector

# class MistralConnector(AbstractConnector):
#     """
#     Chat model implementation for the Mistral API.
#     """

#     def __init__(self, provider_model: str):
#         self.api_key = os.getenv("API_KEY_MISTRAL")
#         if not self.api_key:
#             raise EnvironmentError("API_KEY_MISTRAL not found in environment variables.")
#         self.provider_model = provider_model
#         self.client = Mistral(api_key=self.api_key)

#     def send_prompt(self, prompt: str) -> str:
#         response = self.client.chat.complete(
#             model=self.provider_model,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response.choices[0].message.content
