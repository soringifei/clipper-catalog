"""Standalone: python -m rebels_highlights.gym --input <folder|file> [...]"""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
