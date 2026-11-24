import argparse
import datetime as dt
from pathlib import Path


def get_time_difference(log_file_path: str) -> float:
    """
    Get the time difference between two timestamps in seconds by reading the test / train log file.

    Args:
        log_file_path (str): The path to the log file.
        start_timestamp (str): The start timestamp.
        end_timestamp (str): The end timestamp.

    Returns:
        float: The time difference in seconds.
    """
    log_file = Path(log_file_path)
    if not log_file.exists():
        raise FileNotFoundError("Log file not found.")

    with open(log_file, "r") as f:
        log_content = f.readlines()[
            -6:-4
        ]  # Read the last 6th and 5th lines from the log file

    # Extract the timestamps from the log lines and convert them to datetime objects
    str_time_1 = str.join(" ", log_content[0].strip().split(" ")[0:2])
    str_time_2 = str.join(" ", log_content[1].strip().split(" ")[0:2])

    time_1 = dt.datetime.fromisoformat(str_time_1)
    time_2 = dt.datetime.fromisoformat(str_time_2)

    return (time_2 - time_1).total_seconds()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get the time difference between two timestamps in a log file."
    )
    parser.add_argument("--log-file", help="The path to the log file.")
    args = parser.parse_args()

    if not args.log_file:
        raise ValueError("Please provide the path to the log file.")

    print(f"Loaded log file: {args.log_file}")
    time_difference = get_time_difference(args.log_file)
    print(f"Time difference: {time_difference} seconds")
