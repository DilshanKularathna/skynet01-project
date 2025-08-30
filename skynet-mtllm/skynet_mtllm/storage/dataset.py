import json

class DatasetLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, prompt, response):
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")

    def load(self, limit=None):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                data = [json.loads(l) for l in lines]
                return data[-limit:] if limit else data
        except FileNotFoundError:
            return []
