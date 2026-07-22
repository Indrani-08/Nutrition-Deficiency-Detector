import os


def create_directory(path):
    """
    Creates directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def print_separator():
    print("=" * 60)


def print_heading(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)