from tomlkit import parse
from pyprojroot import here

config_file_path = here("evals/config.toml")

with open(config_file_path, 'r') as config_file:
    config_content = config_file.read()

config = parse(config_content)
