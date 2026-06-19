import os
import json
import logging
from typing import Any
from pydantic import BaseModel, Field, field_validator, ValidationError

from ..ai_apis.providers import ProviderData
from ..ai_apis.api_types import LLMRequestParams, Prompt
from ..util.environment_vars import parse_api_key_in_config

class FalImageGenModuleConfig(BaseModel):
    enabled: bool = False
    model_name: str = ""
    n_images: int = 1
    allow_nsfw: bool = False
    api_key: str = ""

    @field_validator("api_key", mode="before")
    @classmethod
    def parse_api_key(cls, raw_key: str) -> str:
        if not raw_key:
            return ""
        return parse_api_key_in_config(raw_key)

class MiscOptions(BaseModel):
    llm_fallbacks: list[str] = Field(default_factory=list)
    only_ping_on_response_finish: bool = False
    enable_personality_rewrite: bool = False
    enable_knowledge_retrieval: bool = False
    enable_knowledge_summarization: bool = False
    remove_trailing_newline: bool = False
    enable_image_viewing: bool = False
    enable_moderation: bool = False
    botname: str = "Reynard"

class MemorySettings(BaseModel):
    full_history_length: int = 100
    enable_long_term_memory: bool = False
    short_term_history_length: int = 50
    enable_medium_term_memory: bool = False
    medium_term_history_length: int = 20

class ProfileDefaultsProvider:
    def get_default_prompts(self) -> dict[str, Prompt]:
        return {
            "PERSONALITY": Prompt(messages=(
                Prompt.system_msg("You are a helpful AI assistant named Reynard.\nKnowledge:\n((knowledge))\nOld memories:\n((old_memories))\nSummary of conversation:\n((summary))\nCurrent time: ((now))"),
            )),
            "PERSONALITY_REWRITE": Prompt(messages=(
                Prompt.system_msg("Rewrite the following message in character:\n((message))"),
            )),
            "USER_QUERY_REPHRASE": Prompt(messages=(
                Prompt.system_msg("Rephrase the following conversation history into a single standalone query for a knowledge base search. Keep it short.\nLast user: ((last_user))\nHistory:\n((user_query))"),
            )),
            "HISTORY_SUMMARIZE": Prompt(messages=(
                Prompt.system_msg("Summarize the following messages concisely:\n((messages))"),
            )),
            "ATTACHMENT_DESCRIBE": Prompt(messages=(
                Prompt.system_msg("Describe the attached image. The user asked: ((query)). Attachment: ((attachment_url))"),
            )),
            "INFO_SELECT": Prompt(messages=(
                Prompt.system_msg("Extract and summarize the relevant info from the available text that helps answer: ((user_query))\nAvailable Info:\n((available_info))"),
            )),
            "MODERATOR": Prompt(messages=(
                Prompt.system_msg("Review the following user message for safety. Output JSON with format {\"flagged\": boolean}:\n((message))"),
            )),
        }

    def get_default_request_params(self) -> dict[str, LLMRequestParams]:
        return {}

    def get_default_providers(self) -> dict[str, ProviderData]:
        api_key: str = os.getenv("OPENAI_API_KEY", "")
        api_base: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        return {
            "openai": ProviderData(provider_name="openai", api_base=api_base, api_key=api_key)
        }

class Profile(BaseModel):
    options: MiscOptions = Field(default_factory=MiscOptions)
    memory_settings: MemorySettings = Field(default_factory=MemorySettings)
    prompts: dict[str, Prompt] = Field(default_factory=dict)
    request_params: dict[str, LLMRequestParams] = Field(default_factory=dict)
    lang: dict[str, str] = Field(default_factory=dict)
    providers: dict[str, ProviderData] = Field(default_factory=dict)
    regex_replacements: dict[str, str | list[str]] = Field(default_factory=dict)
    fal_image_gen_config: FalImageGenModuleConfig = Field(default_factory=FalImageGenModuleConfig)

    def get_provider_by_name(self, target_name: str) -> ProviderData:
        if target_name in self.providers:
            return self.providers[target_name]
        for provider_name, provider_data in self.providers.items():
            if provider_name.lower() == target_name.lower():
                return provider_data
        raise RuntimeError(f"Failed to get provider '{target_name}'")

    def get_prompt_by_name(self, target_name: str) -> Prompt:
        if target_name in self.prompts:
            return self.prompts[target_name]
        raise RuntimeError(f"Failed to get prompt '{target_name}'")

    def get_request_params_by_name(self, target_name: str) -> LLMRequestParams:
        if target_name in self.request_params:
            return self.request_params[target_name]
        raise RuntimeError(f"Failed to get parameter set named '{target_name}'")

class JsonFileReader:
    def read_dictionary_from_file(self, file_path: str) -> dict[str, Any] | None:
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file_handle:
                return json.load(file_handle)
        except Exception as exception:
            logging.error(f"Error loading JSON from {file_path}, aborting")
            raise exception

class ProfileLoader:
    def __init__(self, file_reader: JsonFileReader, defaults_provider: ProfileDefaultsProvider) -> None:
        self.file_reader: JsonFileReader = file_reader
        self.defaults_provider: ProfileDefaultsProvider = defaults_provider

    def merge_data_with_defaults(self, loaded_data: dict[str, Any]) -> dict[str, Any]:
        merged_prompts = self.defaults_provider.get_default_prompts()
        if "prompts" in loaded_data:
            merged_prompts.update(loaded_data["prompts"])
        loaded_data["prompts"] = merged_prompts

        merged_params = self.defaults_provider.get_default_request_params()
        if "request_params" in loaded_data:
            merged_params.update(loaded_data["request_params"])
        loaded_data["request_params"] = merged_params

        merged_providers = self.defaults_provider.get_default_providers()
        if "providers" in loaded_data:
            merged_providers.update(loaded_data["providers"])
        loaded_data["providers"] = merged_providers

        return loaded_data

    def load_profile_from_directory(self, directory_path: str) -> Profile:      
        raw_data: dict[str, Any] = {}

        options_data = self.file_reader.read_dictionary_from_file(os.path.join(directory_path, "options.json"))
        if options_data:
            raw_data["options"] = options_data

        memory_data = self.file_reader.read_dictionary_from_file(os.path.join(directory_path, "memory.json"))
        if memory_data:
            raw_data["memory_settings"] = memory_data

        prompts_data = self.file_reader.read_dictionary_from_file(os.path.join(directory_path, "prompts.json"))
        if prompts_data:
            prompts_dict: dict[str, Prompt] = {}
            for prompt_name, prompt_config in prompts_data.items():
                if isinstance(prompt_config, dict):
                    prompts_dict[prompt_name] = Prompt(**prompt_config)
                else:
                    prompts_dict[prompt_name] = prompt_config
            raw_data["prompts"] = prompts_dict

        params_data = self.file_reader.read_dictionary_from_file(os.path.join(directory_path, "request_params.json"))
        if params_data:
            params_dict: dict[str, LLMRequestParams] = {}
            for param_name, param_config in params_data.items():
                if isinstance(param_config, dict):
                    params_dict[param_name] = LLMRequestParams(**param_config)
                else:
                    params_dict[param_name] = param_config
            raw_data["request_params"] = params_dict

        lang_data = self.file_reader.read_dictionary_from_file(os.path.join(directory_path, "lang.json"))
        if lang_data:
            raw_data["lang"] = lang_data

        providers_data = self.file_reader.read_dictionary_from_file(os.path.join(directory_path, "providers.json"))
        if providers_data:
            providers_dict: dict[str, ProviderData] = {}
            for provider_name, provider_config in providers_data.items():
                if isinstance(provider_config, dict):
                    providers_dict[provider_name] = ProviderData(**provider_config)
                else:
                    providers_dict[provider_name] = provider_config
            raw_data["providers"] = providers_dict

        regex_data = self.file_reader.read_dictionary_from_file(os.path.join(directory_path, "regex_replacements.json"))
        if regex_data:
            raw_data["regex_replacements"] = regex_data

        image_data = self.file_reader.read_dictionary_from_file(os.path.join(directory_path, "fal_image_gen.json"))
        if image_data:
            raw_data["fal_image_gen_config"] = image_data

        final_data = self.merge_data_with_defaults(raw_data)
        return Profile.model_validate(final_data)

    def load_profile_from_file(self, file_path: str) -> Profile:
        try:
            with open(file_path, 'r', encoding='utf-8') as file_handle:
                raw_data = json.load(file_handle)
        except FileNotFoundError as exception:
            logging.error(f"Profile file not found: {file_path}")
            raise exception
        except json.JSONDecodeError as exception:
            logging.error(f"Error decoding JSON from profile file {file_path}: {exception}")
            raise exception
        except Exception as exception:
            logging.error(f"Error creating Profile instance from file {file_path}: {exception}")
            raise exception

        try:
            final_data = self.merge_data_with_defaults(raw_data)
            return Profile.model_validate(final_data)
        except ValidationError as exception:
            self._log_validation_errors(exception, file_path)
            raise RuntimeError(f"Failed to parse {file_path}") from exception

    def save_profile_to_file(self, profile: Profile, output_path: str) -> None:
        profile_dict = profile.model_dump(mode='json', by_alias=True)
        with open(output_path, 'w', encoding='utf-8') as file_handle:
            json.dump(profile_dict, file_handle, indent=2)
        logging.info(f"Profile saved to {output_path}")

    def _log_validation_errors(self, validation_error: ValidationError, file_path: str) -> None:
        for error in validation_error.errors():
            error_location = error['loc']
            field_path = ".".join([str(field_repr) for field_repr in error_location])
            
            if len(error_location) == 1:
                field_path += " (on the top level of the JSON)"
                
            logging.error(f"Failed to parse field {field_path}: {error['msg']}. (error code: {error['type']})")
