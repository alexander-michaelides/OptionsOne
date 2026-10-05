import asyncio
import argparse

from historical_test import run_historical


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", metavar="SESSION_ID")
    args = parser.parse_args()
    asyncio.run(run_historical(args.resume))
