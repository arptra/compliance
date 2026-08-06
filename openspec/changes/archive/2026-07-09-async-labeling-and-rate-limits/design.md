# Design

Frontend concurrency workers invoked independent row requests, while transport-level retry logic parsed rate-limit responses. This first iteration did not yet provide the final shared serial-to-parallel queue semantics.
