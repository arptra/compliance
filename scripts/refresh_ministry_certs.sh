#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERTS_DIR="${ROOT_DIR}/certs"

ROOT_CERT_URL="${ROOT_CERT_URL:-https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt}"
SUB_CERT_URL="${SUB_CERT_URL:-https://gu-st.ru/content/lending/russian_trusted_sub_ca_pem.crt}"

ROOT_CERT_PATH="${CERTS_DIR}/russian_trusted_root_ca_pem.crt"
SUB_CERT_PATH="${CERTS_DIR}/russian_trusted_sub_ca_pem.crt"
CA_BUNDLE_PATH="${CERTS_DIR}/ca.pem"

mkdir -p "${CERTS_DIR}"

curl -LkfsS "${ROOT_CERT_URL}" -o "${ROOT_CERT_PATH}"
curl -LkfsS "${SUB_CERT_URL}" -o "${SUB_CERT_PATH}"

tmp_root="$(mktemp)"
tmp_sub="$(mktemp)"
tmp_bundle="$(mktemp)"
trap 'rm -f "${tmp_root}" "${tmp_sub}" "${tmp_bundle}"' EXIT

awk '{ sub(/\r$/, ""); print }' "${ROOT_CERT_PATH}" > "${tmp_root}"
awk '{ sub(/\r$/, ""); print }' "${SUB_CERT_PATH}" > "${tmp_sub}"

mv "${tmp_root}" "${ROOT_CERT_PATH}"
mv "${tmp_sub}" "${SUB_CERT_PATH}"

cat "${ROOT_CERT_PATH}" > "${tmp_bundle}"
printf '\n' >> "${tmp_bundle}"
cat "${SUB_CERT_PATH}" >> "${tmp_bundle}"
printf '\n' >> "${tmp_bundle}"
mv "${tmp_bundle}" "${CA_BUNDLE_PATH}"

echo "Updated:"
echo "  ${ROOT_CERT_PATH}"
echo "  ${SUB_CERT_PATH}"
echo "  ${CA_BUNDLE_PATH}"
