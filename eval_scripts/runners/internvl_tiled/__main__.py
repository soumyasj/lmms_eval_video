"""`python -m internvl_tiled <lmms_eval args>`: lmms_eval's CLI with the
`internvl_hf_tiled` model registered (see __init__)."""

import internvl_tiled  # noqa: F401  (registers the model)

from lmms_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
