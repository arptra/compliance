# Design

A process-shared `AdaptiveRateLimiter` issues leases per API endpoint, tracks active and queued requests under a condition lock, honors retry boundaries, and switches serial/parallel modes based on observed responses. Thread-pool row workers remain parallel outside limiter-enforced recovery.
