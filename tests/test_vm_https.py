"""VM HTTPS smoke tests. Run with Python 3.11+: python -m unittest discover -s tests -p test_vm_https.py."""
from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


def clean_env(**overrides: str) -> dict[str, str]:
    prefixes = ("TLS_", "HTTPS_ENABLED", "PUBLIC_", "VITE_", "API_", "DASHBOARD_", "RESOLVED_", "VM_HOSTNAME")
    return {**{key: value for key, value in os.environ.items() if not key.startswith(prefixes)}, **overrides}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class VMRuntimeTests(unittest.TestCase):
    def resolve(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", 'set -euo pipefail; source "$1"; setup_vm_runtime_env; env',
             "test", str(ROOT / "scripts/resolve_runtime_host.sh")],
            env=clean_env(PUBLIC_HOST="vm.example.com", API_PORT="18000", DASHBOARD_PORT="15173", **overrides),
            text=True, capture_output=True, check=False,
        )

    def test_https_is_default_with_secure_urls_and_hmr(self):
        result = self.resolve()
        self.assertEqual(result.returncode, 0, result.stderr)
        values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        self.assertEqual(values["HTTPS_ENABLED"], "1")
        self.assertEqual(values["API_DISPLAY_URL"], "https://vm.example.com:18000")
        self.assertEqual(values["VITE_API_BASE_URL"], values["API_DISPLAY_URL"])
        self.assertEqual(values["DASHBOARD_DISPLAY_URL"], "https://vm.example.com:15173")
        self.assertEqual(values["VITE_PUBLIC_ORIGIN"], values["DASHBOARD_DISPLAY_URL"])
        self.assertEqual(values["VITE_HMR_PROTOCOL"], "wss")

    def test_http_requires_explicit_opt_out(self):
        result = self.resolve(HTTPS_ENABLED="0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("API_DISPLAY_URL=http://vm.example.com:18000", result.stdout)
        self.assertIn("VITE_HMR_PROTOCOL=ws", result.stdout)

    def test_conflicting_or_invalid_settings_fail(self):
        for overrides in (
            {"HTTPS_ENABLED": "yes"}, {"PUBLIC_SCHEME": "http"},
            {"HTTPS_ENABLED": "0", "PUBLIC_SCHEME": "https"},
            {"VITE_API_BASE_URL": "http://vm.example.com:18000"}, {"VITE_HMR_PROTOCOL": "ws"},
        ):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(self.resolve(**overrides).returncode, 0)


@unittest.skipUnless(shutil.which("openssl"), "openssl is required to create test certificates")
class HTTPSCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="compliance-https-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)

        def openssl(*args: str):
            subprocess.run(["openssl", *args], cwd=cls.directory, capture_output=True, check=True)

        openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2", "-subj", "/CN=Test Root",
                "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign",
                "-keyout", "root.key", "-out", "root.pem")
        for name, subject, issuer in (("intermediate", "Test Intermediate", "root"), ("server", "localhost", "intermediate")):
            openssl("req", "-new", "-newkey", "rsa:2048", "-nodes", "-subj", f"/CN={subject}",
                    "-keyout", f"{name}.key", "-out", f"{name}.csr")
            extensions = (
                "basicConstraints=critical,CA:TRUE,pathlen:0\nkeyUsage=critical,keyCertSign,cRLSign\n"
                if name == "intermediate" else
                "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
                "extendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost,IP:127.0.0.1\n"
            )
            (cls.directory / f"{name}.ext").write_text(extensions)
            openssl("x509", "-req", "-in", f"{name}.csr", "-CA", f"{issuer}.pem", "-CAkey", f"{issuer}.key",
                    "-CAcreateserial", "-days", "2", "-extfile", f"{name}.ext", "-out", f"{name}.pem")
        cls.fullchain = cls.directory / "fullchain.pem"
        cls.fullchain.write_bytes((cls.directory / "server.pem").read_bytes() + (cls.directory / "intermediate.pem").read_bytes())

    def prepare(self, output: Path, cert: Path | None = None, key: Path | None = None, chain: Path | None = None):
        args = [sys.executable, str(ROOT / "scripts/prepare_https.py"), "--cert", str(cert or self.fullchain),
                "--key", str(key or self.directory / "server.key"), "--output", str(output)]
        if chain is not None:
            args.extend(["--chain", str(chain)])
        return subprocess.run(args, env=clean_env(), capture_output=True, text=True, check=False)

    def test_ready_fullchain_and_separate_intermediates_produce_same_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, cert, chain in (
                ("ready", self.fullchain, None),
                ("separate", self.directory / "server.pem", self.directory / "intermediate.pem"),
            ):
                with self.subTest(name=name):
                    output = Path(temporary) / name / "fullchain.pem"
                    result = self.prepare(output, cert=cert, chain=chain)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(output.read_bytes(), self.fullchain.read_bytes())

    def test_bad_key_or_chain_does_not_replace_previous_fullchain(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fullchain.pem"
            output.write_text("previous chain")
            malformed = Path(temporary) / "malformed.pem"
            malformed.write_text("-----BEGIN CERTIFICATE-----\ninvalid\n-----END CERTIFICATE-----\n")
            for overrides in (
                {"key": self.directory / "root.key"}, {"chain": malformed},
                {"cert": Path(temporary) / "missing.pem"},
            ):
                with self.subTest(overrides=overrides):
                    self.assertNotEqual(self.prepare(output, **overrides).returncode, 0)
                    self.assertEqual(output.read_text(), "previous chain")

    def test_vm_entrypoints_validate_before_launch_or_stop(self):
        with tempfile.TemporaryDirectory(prefix="vm with spaces-") as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            for name in ("start_vm.sh", "start_vm_https.sh", "restart_vm.sh", "resolve_runtime_host.sh", "setup_https_env.sh", "prepare_https.py"):
                shutil.copy2(ROOT / "scripts" / name, scripts / name)
            python = root / ".venv311/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            for name in ("start_local_stack.sh", "restart_local_stack.sh"):
                launcher = scripts / name
                launcher.write_text('#!/usr/bin/env bash\necho "DISPATCH:${HTTPS_ENABLED}:${PUBLIC_SCHEME}"\n')
                launcher.chmod(0o755)
            certificates = root / "certs/server"
            certificates.mkdir(parents=True)
            for name in ("start_vm.sh", "restart_vm.sh", "start_vm_https.sh"):
                with self.subTest(entrypoint=name):
                    result = subprocess.run(["bash", str(scripts / name)], cwd="/", env=clean_env(PUBLIC_HOST="localhost"),
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("DISPATCH", result.stdout)
                    self.assertIn("HTTPS is enabled", result.stderr)
            shutil.copy2(self.fullchain, certificates / "fullchain.pem")
            shutil.copy2(self.directory / "server.key", certificates / "privkey.pem")
            for name in ("start_vm.sh", "restart_vm.sh", "start_vm_https.sh"):
                with self.subTest(entrypoint=name):
                    result = subprocess.run(["bash", str(scripts / name)], cwd="/", env=clean_env(PUBLIC_HOST="localhost"),
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("DISPATCH:1:https", result.stdout)

    @unittest.skipUnless(shutil.which("node") and (ROOT / "apps/dashboard/node_modules/vite").is_dir(), "Vite dependencies are required")
    def test_api_dashboard_and_proxy_serve_https_with_full_chain(self):
        api_port, dashboard_port = free_port(), free_port()
        env = clean_env(
            HTTPS_ENABLED="1", TLS_CERT_FILE=str(self.fullchain), TLS_KEY_FILE=str(self.directory / "server.key"),
            TLS_CA_FILE=str(self.directory / "root.pem"), API_HOST="127.0.0.1", API_PORT=str(api_port),
            PYTHONPATH=str(ROOT / "src"), RUNTIME_CONFIG=str(ROOT / "configs/project.yaml"),
            VITE_API_BASE_URL=f"https://127.0.0.1:{api_port}", VITE_HMR_PROTOCOL="wss",
        )
        context = ssl.create_default_context(cafile=str(self.directory / "root.pem"))

        def get(port: int, path: str):
            connection = http.client.HTTPSConnection("127.0.0.1", port, context=context, timeout=1)
            try:
                connection.request("GET", path)
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        with tempfile.TemporaryDirectory() as temporary:
            with open(Path(temporary) / "servers.log", "w+") as log:
                processes = []
                try:
                    api = subprocess.Popen([sys.executable, str(ROOT / "scripts/start_api.py")], env=env, cwd=ROOT,
                                           stdout=log, stderr=log)
                    processes.append(api)
                    vite_url = (ROOT / "apps/dashboard/node_modules/vite/dist/node/index.js").as_uri()
                    options = {
                        "root": str(ROOT / "apps/dashboard"), "configFile": str(ROOT / "apps/dashboard/vite.config.ts"),
                        "cacheDir": str(Path(temporary) / "vite-cache"),
                        "server": {"host": "127.0.0.1", "port": dashboard_port, "strictPort": True},
                    }
                    vite_code = f"import {{createServer}} from {json.dumps(vite_url)}; await (await createServer({json.dumps(options)})).listen();"
                    dashboard = subprocess.Popen(["node", "--input-type=module", "-e", vite_code], env=env, cwd=ROOT,
                                                 stdout=log, stderr=log)
                    processes.append(dashboard)
                    for process, port in ((api, api_port), (dashboard, dashboard_port)):
                        deadline = time.monotonic() + 30
                        while time.monotonic() < deadline:
                            if process.poll() is not None:
                                break
                            try:
                                status, body = get(port, "/api/health")
                                if status == 200:
                                    self.assertEqual(json.loads(body)["status"], "ok")
                                    break
                            except OSError:
                                pass
                            time.sleep(0.1)
                        else:
                            self.fail(f"HTTPS did not become ready on port {port}")
                        self.assertIsNone(process.poll(), "HTTPS server exited unexpectedly")
                    self.assertEqual(get(dashboard_port, "/")[0], 200)
                    status, hmr = get(dashboard_port, "/@vite/client")
                    self.assertEqual(status, 200)
                    self.assertIn(b"wss", hmr)
                    # HTTP must not serve content on either TLS listener.
                    for port in (api_port, dashboard_port):
                        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
                        try:
                            connection.request("GET", "/api/health")
                            try:
                                self.assertGreaterEqual(connection.getresponse().status, 400)
                            except (OSError, http.client.HTTPException):
                                pass
                        finally:
                            connection.close()
                except Exception:
                    log.flush()
                    log.seek(0)
                    print(log.read(), file=sys.stderr)
                    raise
                finally:
                    for process in processes:
                        process.terminate()
                    for process in processes:
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()


if __name__ == "__main__":
    unittest.main()
