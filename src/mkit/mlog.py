import logging
import numpy as np

def get_logger(file_name, file_level=logging.INFO, console_level=logging.DEBUG):
    """
    Create a custom logger using for both file and console
    Parameters
    ----------
    file_name : string
        Name of log file and also name of logger
    file_level : logging.level (optional, default = None)
        Logging level for log to file
    console_level : logging.level (optional, default = None)
        Logging level for log to console
    Returns
    -------
    logger : logger
        A logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if console_level != None:
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch_format = logging.Formatter('%(asctime)s -  %(funcName)s - %(message)s')
        ch.setFormatter(ch_format)
        logger.addHandler(ch)
    if file_level != None:
        fh = logging.FileHandler(file_name)
        fh.setLevel(file_level)
        fh_format = logging.Formatter('%(asctime)s - %(funcName)s -%(message)s')
        fh.setFormatter(fh_format)
        logger.addHandler(fh)
    return logger

def config():
    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(linewidth=300)