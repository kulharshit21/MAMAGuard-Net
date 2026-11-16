#!/usr/bin/env python3
from mamaguardnet.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["audit-data", *__import__("sys").argv[1:]]))

