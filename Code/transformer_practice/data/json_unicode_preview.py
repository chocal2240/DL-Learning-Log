"""Preview JSON data with readable Chinese text."""

from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_JSON_PATH = DATA_DIR / "json" / "test.json"


def preview_json(path: str | Path = DEFAULT_JSON_PATH, limit: int = 3) -> None:
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print(json.dumps(data[:limit], ensure_ascii=False, indent=2))
    print(type(data), len(data), type(data[0]), len(data[0]))


if __name__ == "__main__":
    preview_json()
