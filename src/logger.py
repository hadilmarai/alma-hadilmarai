import json
import time
from typing import Any, Dict

class JSONLLogger:
    def __init__(self, path: str):
        self.path = path

    def log(self, event_type: str, payload: Dict[str, Any]):
        record = {
            "ts": time.time(),
            "event_type": event_type,
            "payload": payload
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
