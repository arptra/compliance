from __future__ import annotations

import os

import uvicorn

from complaints_trends.api.app import create_app


def main() -> None:
    https_enabled = os.environ.get("HTTPS_ENABLED", "0")
    if https_enabled not in {"0", "1"}:
        raise ValueError("HTTPS_ENABLED must be 1 or 0")
    ssl_options = {}
    if https_enabled == "1":
        cert_file = os.environ.get("TLS_CERT_FILE")
        key_file = os.environ.get("TLS_KEY_FILE")
        if not cert_file or not key_file:
            raise ValueError("HTTPS requires TLS_CERT_FILE (fullchain) and TLS_KEY_FILE")
        ssl_options = {
            "ssl_certfile": cert_file,
            "ssl_keyfile": key_file,
            "ssl_keyfile_password": os.environ.get("TLS_KEY_PASSWORD", ""),
        }
    config_path = os.environ.get("RUNTIME_CONFIG", "/tmp/project.runtime.yaml")
    app = create_app(config_path)
    uvicorn.run(
        app,
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        log_level="info",
        **ssl_options,
    )


if __name__ == "__main__":
    main()
