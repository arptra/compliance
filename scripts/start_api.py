from __future__ import annotations

import os

import uvicorn

from complaints_trends.api.app import create_app


def main() -> None:
    config_path = os.environ.get("RUNTIME_CONFIG", "/tmp/project.runtime.yaml")
    app = create_app(config_path)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", "8000")), log_level="info")


if __name__ == "__main__":
    main()
