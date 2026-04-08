from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("flowMC")
except PackageNotFoundError:
    __version__ = "unknown"
