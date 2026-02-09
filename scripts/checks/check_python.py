import platform
import sys


def _version_or_error(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        version = getattr(mod, "__version__", "unknown")
        return f"{module_name}: {version}"
    except Exception as exc:
        return f"{module_name}: import error ({exc})"


def main() -> None:
    print("=== Python Runtime Info ===")
    print(f"Executable: {sys.executable}")
    print(f"Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Implementation: {platform.python_implementation()}")
    if hasattr(sys, "_is_gil_enabled"):
        print(f"GIL Enabled: {sys._is_gil_enabled()}")
    print(_version_or_error("numpy"))
    print(_version_or_error("pandas"))


if __name__ == "__main__":
    main()
