from app.compare_december import run_compare_december

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Backward-compatible alias for stage-3 december compare")
    p.add_argument("--config", required=True)
    a = p.parse_args()
    run_compare_december(a.config)
