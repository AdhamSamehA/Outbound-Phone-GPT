import logging
import time

class MyFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        t = time.strftime("%H:%M:%S", ct)
        s = "%s,%03d" % (t, record.msecs)
        return s

def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if logger already has handlers
    if not logger.handlers:
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = MyFormatter(log_format)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.propagate = False  # Prevent the log messages from being propagated to the root logger

    return logger

