# Design

The change removed registrations rather than deleting every legacy module. Therefore file existence alone is not proof of a public route; agents must verify `api/app.py` and `routes/index.tsx`.
