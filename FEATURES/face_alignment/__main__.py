"""CLI entrypoint for face alignment - proxies to src.__main__"""
from .src.__main__ import main
import sys

if __name__ == "__main__":
    sys.exit(main())
