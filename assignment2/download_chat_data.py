"""
Deprecated entry point kept for compatibility.

The corpus is now built from the imported batches under data/raw by
build_chat_data.py.
"""

from build_chat_data import main


if __name__ == "__main__":
    main()
