"""The classifier requires a key at construction; no request is ever sent."""

import os

os.environ.setdefault("TYPESAFE_API_KEY", "test-key")
