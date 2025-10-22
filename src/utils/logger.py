import logging
import os

def get_logger(name: str = "art_lora", log_dir: str = "logs", log_file: str = "art_lora.log"):
    # create a log directory if it does not exists
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name)
    if not logger.handlers:  # taking care of duplicate handler
        logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_path, mode='a')
        file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger
