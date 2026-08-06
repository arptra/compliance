# Design

Batch size was constrained by tokens, incomplete responses retried only missing records, and parallel workers shared thread-safe persistence. Full-dialog context remained the model input boundary.
