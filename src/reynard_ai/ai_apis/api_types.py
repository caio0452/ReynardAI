import re

from typing import ClassVar, Optional
from pydantic import BaseModel, Field

OpenAIMessage = dict[str, list | str | dict]

class Prompt(BaseModel, frozen=True):
    PROMPT_PLACEHOLDER_RE: ClassVar = re.compile(pattern=r"\(\((\w+)\)\)")
    messages: tuple[OpenAIMessage, ...] = Field(...)

    @staticmethod
    def system_msg(content: str) -> OpenAIMessage:
        return {"role": "system", "content": content}

    def plus(self, message: OpenAIMessage):
        return Prompt(messages=self.messages + (message,))

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

    def replace_or_throw(self, replacements: dict[str, str], *, require_all: bool = True):
        return self.replace(replacements, require_all=require_all, forbid_extra=True)
    
    def replace(self, replacements: dict[str, str], *, require_all: bool = True, forbid_extra: bool = False) -> "Prompt":
        found: set[str] = set()
        missing: set[str] = set()

        def replace_in_str(s: str) -> str:
            def _sub(m: re.Match[str]) -> str:
                key = m.group(1)
                found.add(key)
                if key in replacements:
                    return str(replacements[key])
                missing.add(key)
                return m.group(0)
            return Prompt.PROMPT_PLACEHOLDER_RE.sub(_sub, s)

        def walk(obj):
            if isinstance(obj, str):
                return replace_in_str(obj)
            if isinstance(obj, list):
                return [walk(x) for x in obj]
            if isinstance(obj, dict):
                return {k: walk(v) for k, v in obj.items()}
            return obj

        new_messages = tuple(walk(m) for m in self.messages)

        if require_all and missing:
            raise ValueError(
                f"Missing placeholder replacement for {sorted(missing)}. "
                f"Provided: {sorted(replacements.keys())}"
            )

        if forbid_extra:
            extra = set(replacements.keys()) - found
            if extra:
                raise ValueError(f"The following replacements must be in your prompt: {sorted(extra)}")
            
        return Prompt(messages=new_messages)

    def to_openai_format(self) -> tuple[OpenAIMessage, ...]:
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