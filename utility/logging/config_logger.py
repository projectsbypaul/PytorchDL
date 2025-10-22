import logging
import sys, os
from logging import Handler

LEVEL_MAP = {
    logging.INFO:    "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR:   "ERROR",
}

class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.levelname = LEVEL_MAP.get(record.levelno, record.levelname)
        return super().format(record)

class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level
    def filter(self, record):
        return record.levelno <= self.max_level

class StreamToLogger:
    """Redirects writes to a logger (for print() capture)."""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self._buffer = ""
    def write(self, buf):
        self._buffer += buf
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.logger.log(self.level, line)
    def flush(self):
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""

def init_log(log_file: str | None = None, capture_print: bool = False):
    # Root logger at DEBUG so DEBUG actually passes through.
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Force=True replaces existing handlers (important if something configured logging earlier).
    logging.basicConfig(level=logging.INFO, handlers=[], force=True)

    root = logging.getLogger()

    # stdout: DEBUG and INFO
    h_out = logging.StreamHandler(sys.stdout)
    h_out.setLevel(logging.DEBUG)
    h_out.addFilter(MaxLevelFilter(logging.INFO))
    h_out.setFormatter(CustomFormatter(fmt, datefmt))
    root.addHandler(h_out)

    # stderr: WARNING and above
    h_err = logging.StreamHandler(sys.stderr)
    h_err.setLevel(logging.WARNING)
    h_err.setFormatter(CustomFormatter(fmt, datefmt))
    root.addHandler(h_err)

    # optional file handler (receives everything)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(CustomFormatter(fmt, datefmt))
        root.addHandler(fh)

    # optionally capture print() -> logging.INFO
    if capture_print:
        sys.stdout = StreamToLogger(logging.getLogger("print"), logging.INFO)
        sys.stderr = StreamToLogger(logging.getLogger("print"), logging.WARNING)

    # Route warnings.warn(...) into logging.WARNING
    logging.captureWarnings(True)

# ---- demo ----
if __name__ == "__main__":
    init_log(log_file=None, capture_print=True)  # set capture_print=False to keep normal prints
    print("hello from print()")                  # goes to INFO via logger if capture_print=True
    logging.debug("debug to stdout")
    logging.info("info to stdout")
    logging.warning("warning to stderr")
    logging.error("error to stderr")
