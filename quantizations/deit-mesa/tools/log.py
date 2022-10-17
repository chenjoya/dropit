import os, sys
import logging.config

def setup_logging(log_file='info.txt', resume=False, dummy=False, stdout=True):
    """
    Setup logging configuration
    """
    if dummy:
        logging.getLogger('dummy')
    else:
        if os.path.isfile(log_file) and resume:
            file_mode = 'a'
        else:
            file_mode = 'w'

        logging.shutdown() # shutdown all logging before
        root_logger = logging.getLogger()
        if root_logger.handlers:
            root_logger.handlers[0].close()
            root_logger.removeHandler(root_logger.handlers[0])
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            filename=log_file,
                            filemode=file_mode)
        if stdout:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            console.setFormatter(formatter)
            logging.getLogger('').addHandler(console)


class text_logger(object):
    def __init__(self, log_file, colored=False):
        ENDC = '\033[0m'
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

        self.log_file = sys.stdout
        if isinstance(log_file, str):
            if os.path.isfile(log_file):
                self.log_file = open(log_file, 'a')
            else:
                self.log_file = open(log_file, 'w')

        elif hasattr(log_file, 'closed') and log_file.closed == False:
            self.log_file = log_file

        self.ic = '' 
        self.wc = '' 
        self.ec = '' 
        self.nc = '' 
        if colored:
            self.ic = OKGREEN
            self.wc = WARNING
            self.ec = FAIL
            self.nc = ENDC

    def info(self, string):
        print("{}info ==>{} {}".format(self.ic, self.nc, string), file=self.log_file)
        print("{}info ==>{} {}".format(self.ic, self.nc, string))

    def warning(self, string):
        print("{}info ==>{} {}".format(self.wc, self.nc, string), file=self.log_file)
        print("{}info ==>{} {}".format(self.wc, self.nc, string))

    def error(self, string):
        print("{}info ==>{} {}".format(self.ec, self.nc, string), file=self.log_file)
        print("{}info ==>{} {}".format(self.ec, self.nc, string))

