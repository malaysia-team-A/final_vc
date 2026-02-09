import sys


def main() -> None:
    print("Python works!")
    print(f"Executable: {sys.executable}")
    print(f"Version: {sys.version.splitlines()[0]}")


if __name__ == "__main__":
    main()
