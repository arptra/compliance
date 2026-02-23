import pytest

from complaints_trends.gigachat_mtls import _validate_mtls_files


def test_validate_mtls_files_raises_for_missing_files():
    with pytest.raises(FileNotFoundError) as exc:
        _validate_mtls_files("missing/ca.pem", "missing/client.pem", "missing/client.key")
    assert "mTLS files are missing" in str(exc.value)
