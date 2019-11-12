"""Utility functions."""

import datetime
import os
import sys
from typing import List, Optional

CLASSES = ["concrete_cement", "healthy_metal", "incomplete", "irregular_metal", "other"]
LOCATIONS = {
    "colombia": ["borde_rural", "borde_soacha"],
    "guatemala": ["mixco_1_and_ebenezer", "mixco_3"],
    "st_lucia": ["castries", "dennery", "gros_islet"],
}


class DualLogger:
    """Logger to write to stdout as well as file."""

    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.log = open(log_filepath, "a")

    def write(self, message):
        """
        Log a message to the terminal as well as the log file.
        :param message: Message to write.
        :return: None.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """
        Flush the logger.
        :return: None.
        """


class UnknownClassException(Exception):
    """Thrown when an unknown class is asked for."""


def get_indexed_class_names() -> List[str]:
    """Get class names with indices."""
    return [str(i) + "_" + class_name for (i, class_name) in enumerate(CLASSES)]


def get_indexed_class_name(class_name: str) -> str:
    """Get a single indexed class name."""
    if class_name in CLASSES:
        return str(CLASSES.index(class_name)) + "_" + class_name
    raise UnknownClassException(f'Unknown class "{class_name}"')


def create_timestamp_str() -> str:
    """Create a timestamp string for the current time."""
    today = datetime.datetime.now()
    return today.strftime("%Y-%m-%d_%X")


def create_dirs_if_not_found(path: str) -> None:
    """
    Create dirs at path if they don't already exist.
    :param path: Path to dirs.
    :return: None.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def count_files_recursive(path: str, contains: Optional[str] = None) -> int:
    """
    Recursively count files in a path.
    :param path: Starting dir.
    :param contains: Directories must contain this string.
    :return: Number of files found.
    """
    file_count = 0
    for dir_name, _, files in os.walk(path):
        if contains is None or contains in dir_name:
            file_count += len(files)
    return file_count


def class_distribution(path: str) -> List[float]:
    """
    Find the class distribution in a directory.
    :param path: Path to dir.
    :return: Class distribution.
    """
    class_dist = []
    for class_name in get_indexed_class_names():
        class_dir = os.path.join(path, class_name)
        class_count = count_files_recursive(class_dir)
        class_dist.append(class_count)
    return class_dist
