"""Assemble and validate the server certificate chain without copying its key."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import ssl
import tempfile


def read_certificates(path: Path) -> list[str]:
    content = path.read_text(encoding="ascii")
    certificates = re.findall(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", content, re.DOTALL
    )
    if not certificates or re.sub(
        r"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", "", content, flags=re.DOTALL
    ).strip():
        raise ValueError(f"Expected only PEM certificates in {path}")
    return certificates


def prepare_https(cert: Path, key: Path, chain: Path | None, output: Path) -> int:
    certificates = read_certificates(cert)
    if chain is not None:
        certificates.extend(read_certificates(chain))
    # Preserve issuer order while avoiding duplicates in overlapping bundles.
    certificates = list(dict.fromkeys(certificates))
    if not key.is_file():
        raise ValueError(f"Private key is missing: {key}")
    if output.resolve() in {path.resolve() for path in (cert, key, chain) if path is not None}:
        raise ValueError("The generated fullchain must not overwrite an input file")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write("\n".join(certificates) + "\n")
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        # An explicit empty password also prevents an interactive prompt on a VM.
        context.load_cert_chain(temporary, key, password=os.environ.get("TLS_KEY_PASSWORD", ""))
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(certificates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cert", required=True, type=Path)
    parser.add_argument("--key", required=True, type=Path)
    parser.add_argument("--chain", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        count = prepare_https(args.cert, args.key, args.chain, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"HTTPS certificate setup failed: {exc}\n")
    print(f"HTTPS certificate/key loaded: {args.output} ({count} certificate(s))")


if __name__ == "__main__":
    main()
