"""Local JSON persistence for holdings, watchlist and settings (data/ folder, git-ignored)."""
import json
import os
import threading

_lock = threading.Lock()
DEFAULT = {"holdings": [], "watchlist": [], "settings": {}}


class Store:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def load(self):
        try:
            with open(self.path) as f:
                d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            d = {}
        return {**json.loads(json.dumps(DEFAULT)), **d}

    def save(self, d):
        with _lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(d, f, indent=2)
            os.replace(tmp, self.path)
