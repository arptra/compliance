import numpy as np

from app.cluster import _top_terms_ctfidf
from app.preprocess import TextPreprocessor


def test_ctfidf_terms_exclude_placeholders_and_garbage():
    cfg = {
        "preprocess": {
            "use_razdel": False,
            "lemmatize": False,
            "extra_stopwords_path": "configs/extra_stopwords.txt",
            "deny_tokens_path": "configs/deny_tokens.txt",
            "remove_placeholders": True,
        }
    }
    pre = TextPreprocessor(cfg)
    texts = [
        "верните деньги списали комиссию",
        "не работает перевод верните деньги",
        "<num> xxxx *** id=abcd1234",
    ]
    labels = np.array([0, 0, 0])
    terms = _top_terms_ctfidf(texts, labels, n_clusters=1, preprocessor=pre, top_n=10)[0]
    joined = " ".join(terms)
    assert "xxxx" not in joined
    assert "<num>" not in joined
    assert "id" not in joined
    assert any("верните" in t or "деньги" in t for t in terms)
