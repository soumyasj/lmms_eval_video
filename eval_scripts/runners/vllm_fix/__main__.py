"""`python -m vllm_fix <lmms_eval args>`: lmms_eval's CLI with the `vllm_fix`
model registered (see __init__)."""

import vllm_fix  # noqa: F401  (registers the model)

from lmms_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
