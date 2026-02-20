from app.train_full import run_train_full

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Backward-compatible alias for stage-2 full training")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", type=int, default=None, help="Ignored in staged mode")
    a = p.parse_args()
    run_train_full(a.config)
