import os


def check(write: bool = False):
    if not os.getenv("DB_URL"):
        raise RuntimeError("DB_URL missing")
    return True
