import logging

class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    base_format = "[%(asctime)s %(levelname)s] %(message)s"

    LEVEL_COLORS = {
        logging.DEBUG: grey,
        logging.INFO: grey,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def __init__(self):
        date_format = "%d/%m/%y %H:%M:%S"
        super().__init__(fmt=self.base_format, datefmt=date_format)

    def format(self, record):
        color_prefix = self.LEVEL_COLORS.get(record.levelno, self.reset)
        formatted_message = super().format(record)
        return color_prefix + formatted_message + self.reset
    
def setup():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(ColoredFormatter())
    root_logger.addHandler(ch)