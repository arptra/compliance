from complaints_trends.taxonomy import load_taxonomy


def test_taxonomy_supports_labels_and_dict_subcategories():
    tx = load_taxonomy("configs/categories_seed.yaml")
    assert "BILLING" in tx["category_codes"]
    assert "duplicate_charge" in tx["subcategories_by_category"]["BILLING"]
    assert "CREDITING" in tx["category_codes"]
    assert "collection_issue" in tx["subcategories_by_category"]["CREDITING"]
    assert "NONE" in tx["loan_products"]
    assert "CONSUMER_LOAN" in tx["loan_products"]
