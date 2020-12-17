import os
import logging
from time import gmtime, strftime

LOGGING_DIR="logs"

LOGGER = logging.getLogger(__name__)


class AutoLogger:
    @staticmethod
    def setup_logging():
        # Setup Logging
        log_name = '{}_{}_{}_{}'.format("datset", "model", "batch_size", strftime("%Y_%m_%d_%H_%M_%S", gmtime()))

        logging.basicConfig(level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=os.path.join(LOGGING_DIR, log_name + '.txt'),
                            filemode='a')

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
