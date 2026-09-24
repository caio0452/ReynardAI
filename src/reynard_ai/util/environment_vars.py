import os
import logging
from dotenv import load_dotenv

load_dotenv()

def get_environment_var(var_name: str, *, required: bool = False) -> str:
    value = os.getenv(var_name)
    if required and (value is None or value == ""):
        logging.warning(f"Missing '{var_name}' environment variable")
        raise ValueError(f"Missing '{var_name}' environment variable")
    return "" if value is None else value

def parse_api_key_in_config(api_key: str | None, *, required: bool = False) -> str:
    if not api_key:
        return ""
    if api_key.startswith('[') and api_key.endswith(']'):
        env_var_name = api_key[1:-1]
        loaded_api_key = get_environment_var(env_var_name, required=required)
        if required and not loaded_api_key:
            raise ValueError(f"Environment variable {env_var_name} not set for API key.")
        return loaded_api_key
    return api_key
