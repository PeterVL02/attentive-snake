"""Start the web app on 127.0.0.1.

    uv run python -m attentive_snake.app [--port 8000] [--runs-dir runs] [--configs-dir configs]
"""

import argparse
from pathlib import Path

import uvicorn

from attentive_snake.app.application import create_app
from attentive_snake.app.paths import default_paths

HOST = "127.0.0.1"  # local only: the API can start processes and write files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--runs-dir", type=Path, help="parent directory of the runs (default: <project>/runs)")
    parser.add_argument(
        "--configs-dir", type=Path, help="per-game default configs and reward schemes (default: <project>/configs)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(default_paths(args.runs_dir, args.configs_dir))
    print(f"Attentive Snake: http://{HOST}:{args.port}/")
    uvicorn.run(app, host=HOST, port=args.port)


if __name__ == "__main__":
    main()
