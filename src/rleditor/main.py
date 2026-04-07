"""CLI entrypoint for the RL Editor application."""

from rleditor.app import run


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
