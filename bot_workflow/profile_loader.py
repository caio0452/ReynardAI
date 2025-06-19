import json
import logging
from typing import Dict, List
from pydantic import BaseModel, Field, field_validator, ValidationError

from ai_apis.providers import ProviderData
from ai_apis.api_types import LLMRequestParams, Prompt
from util.environment_vars import parse_api_key_in_config

class FalImageGenModuleConfig(BaseModel):
    enabled: bool
    model_name: str
    n_images: int
    allow_nsfw: bool
    api_key: str

    @field_validator("api_key")
    @classmethod
    def parse_api_key(cls, v):
        return parse_api_key_in_config(v)

class Parameters(BaseModel):
    botname: str
    recent_message_history_length: int
    only_ping_on_response_finish: bool
    enable_personality_rewrite: bool
    enable_knowledge_retrieval: bool
    enable_long_term_memory: bool
    remove_trailing_newline: bool
    enable_image_viewing: bool
    llm_fallbacks: List[str] = Field(default_factory=list, examples=["test", "aaa"])

class Profile(BaseModel):
    options: Parameters
    prompts: Dict[str, Prompt]
    request_params: Dict[str, LLMRequestParams]
    lang: Dict[str, str]
    providers: Dict[str, ProviderData]
    regex_replacements: Dict[str, str | list[str]]
    fal_image_gen_config: FalImageGenModuleConfig

    def get_prompt(self, name: str) -> Prompt:
        if name in self.prompts:
            return self.prompts[name].model_copy(deep=True)
        else:
            raise ValueError(f"Request prompt '{name}' does not exist")

    @classmethod
    def from_file(cls, filename: str) -> "Profile":
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logging.error(f"Profile file not found: {filename}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from profile file {filename}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error creating Profile instance from file {filename}: {e}")
            raise

        try:
            return cls.model_validate(data)
        except ValidationError as ex:
            for error in ex.errors():
                error_loc = error['loc']
                field = ".".join([str(field_repr) for field_repr in error_loc])
                if len(error_loc) == 1:
                    field += " (on the top level of the JSON)"
                logging.error(
                    f"Failed to parse field {field}: {error['msg']}. (error code: {error['type']})"
                )
        raise RuntimeError(f"Failed to parse {filename}")

    def save_to_file(self, output_path: str):
        profile_dict = self.model_dump(mode='json', by_alias=True) 
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(profile_dict, f, indent=2)
        logging.info(f"Profile saved to {output_path}")