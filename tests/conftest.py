"""Set env vars before any test module loads torch."""

import os

os.environ["PYTORCH_MPS_DISABLE"] = "1"
