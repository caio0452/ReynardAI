import json
import openai

from typing import Any
from abc import ABC, abstractmethod
from ..ai_apis.providers import ProviderData
from ..ai_apis.api_types import LLMRequestParams, Prompt

class ContentModerator(ABC):
    @abstractmethod
    async def is_flagged(self, input: Any) -> bool:
        raise NotImplementedError("is_flagged")

class OpenAIModerator(ContentModerator):
    def __init__(self, client: openai.AsyncClient):
        self.client = client

    async def is_flagged(self, input) -> Any:
        response = await self.client.moderations.create(
            input=input
        )
        return response.results[0].flagged

# TODO: should be provider (e.g. OpenAI) agnostic
class EmbeddingsClient:
    def __init__(self, provider: ProviderData, model_name: str):
        self.client = openai.AsyncOpenAI(
            api_key=provider.api_key, 
            base_url=provider.api_base
        )
        self.model_name = model_name

    async def vectorize(self, input: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(input, str):
            response = await self.client.embeddings.create(
                input=input,
                model=self.model_name
            )
            return response.data[0].embedding
        elif isinstance(input, list):
            response = await self.client.embeddings.create(
                input=input,
                model=self.model_name
            )
            return [e.embedding for e in response.data]

class SyncEmbeddingsClient:
    def __init__(self, provider: ProviderData):
        self.openai_client = openai.OpenAI(
            api_key=provider.api_key, 
            base_url=provider.api_base
        )

    def vectorize(self, input: str | list[str], model="text-embedding-3-large") -> list[float] | list[list[float]]:
        if isinstance(input, str):
            response = self.openai_client.embeddings.create(
                input=input,
                model=model
            )
            return response.data[0].embedding
        elif isinstance(input, list):
            response = self.openai_client.embeddings.create(
                input=input,
                model=model
            )
            return [e.embedding for e in response.data]

class LLMClient:
    def __init__(self, client: openai.AsyncClient):
        self.client = client

    @classmethod
    def from_openai_client(cls, client: openai.AsyncClient):
        return cls(client)

    @classmethod
    def from_provider(cls, provider: ProviderData):
        client = openai.AsyncOpenAI(
            api_key=provider.api_key, 
            base_url=provider.api_base,
            timeout=15
        )
        return cls.from_openai_client(client)

    async def send_request(self, *, prompt: Prompt, params: LLMRequestParams):
        raw_response = await self.client.chat.completions.create(
            messages=prompt.to_openai_format(), # type: ignore
            model=params.model_name,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            logit_bias=params.logit_bias
        )
        
        if raw_response.choices is None or len(raw_response.choices) == 0:
            resp_json = json.loads(raw_response.to_json())
            if "error" in resp_json and resp_json["error"]:
                raise RuntimeError(resp_json["error"])
            else:
                raise RuntimeError(f"ProviderData returned no response choices. Response was {str(raw_response)}")
        else:
            return raw_response.choices[0]
