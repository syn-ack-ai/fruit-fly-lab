"""
Shared secret for the Habitat server <-> brain client socket.

multiprocessing.connection unpickles what it receives, so the key must not be
guessable by other local users: it is $FLY_HABITAT_KEY if set, otherwise a
random key kept in ~/.fly_habitat_key (created with mode 600). Standard library
only (the server runs in the Habitat conda env).
"""
from __future__ import annotations

import os
import secrets

PATH = os.path.expanduser("~/.fly_habitat_key")


def authkey() -> bytes:
    env = os.environ.get("FLY_HABITAT_KEY")
    if env:
        return env.encode()
    try:
        with open(PATH, "rb") as fh:
            key = fh.read().strip()
        if key:
            return key
    except FileNotFoundError:
        pass
    key = secrets.token_hex(32).encode()
    fd = os.open(PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key)
    return key
