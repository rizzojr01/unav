#!/usr/bin/env python3
"""Backward-compatible entrypoint for `python -m unav.run_mapping`.

Canonical entrypoint is `python -m unav.run_mapper`.
"""

from unav.run_mapper import main


if __name__ == "__main__":
    main()
