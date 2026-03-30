import os
import logging

class CustomLogger:
    @staticmethod
    def get_logger(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        log_file_path = os.path.join(base_dir, "model.log")

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # default level

        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file_path)

        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)

        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        return logger