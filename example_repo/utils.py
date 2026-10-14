"""Small utility module used to test the doc generator's parser."""


def parse_config(path: str) -> dict:
    data = load_file(path)
    return normalize(data)


def load_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def normalize(data: str) -> dict:
    """Already documented — parser should detect this and skip it."""
    return {"raw": data.strip()}


class ConfigManager:
    def __init__(self, path: str):
        self.path = path
        self.config = parse_config(path)

    def reload(self):
        self.config = parse_config(self.path)

    def get(self, key: str):
        return self.config.get(key)
