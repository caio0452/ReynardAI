import re
import json

from typing import Optional
from pydantic import BaseModel, Field

OpenAIMessage = dict[str, list | str | dict]

class Prompt(BaseModel, frozen=True):
    messages: list[OpenAIMessage] = Field(...)

    @staticmethod
    def system_msg(content: str) -> OpenAIMessage:
        return {"role": "system", "content": content}

    def plus(self, message: OpenAIMessage):
        return Prompt(messages=self.messages + [message])

    @staticmethod
    def user_msg(content: str, image_url: str | None = None) -> OpenAIMessage:
        if image_url:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        else:
            return {"role": "user", "content": content}

    @staticmethod
    def assistant_msg(content: str) -> OpenAIMessage:
        return {"role": "assistant", "content": content}

    # TODO: don't hardcode format
    def replace(self, replacements: dict[str, str]) -> "Prompt":
        placeholder_format = "((placeholder))"
        modified_messages = []
        prompt_as_str = json.dumps(self.messages)
        all_formatted_placeholders = [
            placeholder_format.replace("placeholder", k) for k, v in replacements.items()
        ]
        for match in re.findall(r"\(\(\w+\)\)", prompt_as_str):
            if match not in all_formatted_placeholders:
                raise ValueError(f"Missing placeholder replacement for '{match}'. Must specify all prompt placeholders, got only: {replacements}")

        def replace_all_in_dict(dict_data: dict, old_str, new_str) -> dict:
            replaced_dict = dict_data.copy()
            for k, v in dict_data.items():
                if isinstance(v, str):
                    replaced_dict[k] = v.replace(old_str, new_str)
                elif isinstance(v, dict):
                    replaced_dict[k] = replace_all_in_dict(replaced_dict, old_str, new_str)
                else:
                    raise ValueError(f"Cannot parse prompt dictionary because one of the keys is not str or dict: {dict_data}")
            return replaced_dict 
    
        for message in self.messages:
            modified_message = message
            for placeholder, replacement in replacements.items():
                formatted_placeholder = placeholder_format.replace("placeholder", placeholder)
                modified_message = replace_all_in_dict(modified_message, formatted_placeholder, replacement)
            modified_messages.append(modified_message)

        return Prompt(messages=modified_messages)

    def to_openai_format(self) -> list[OpenAIMessage]:
        return self.messages

class LLMRequestParams(BaseModel, frozen=True):
    model_name: str
    temperature: float = 0.5
    max_tokens: int = 300
    logit_bias: Optional[dict[str, int]] = {}

    class Config:
        json_encoders = {
            Prompt: lambda p: p.to_openai_format()
        }