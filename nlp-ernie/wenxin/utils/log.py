"""Arguments for logging."""

import os
import logging
import logging.handlers
import sys


def init_log(log_path, level=logging.INFO, when="D", backup=7,
             format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
             datefmt="%m-%d %H:%M:%S"):
    """
    init_log - initialize log module
    Args:
    log_path - Log file path prefix.
    Log data will go to two files: log_path.log and log_path.log.wf
    Any non-exist parent directories will be created automatically
    level - msg above the level will be displayed
    DEBUG < INFO < WARNING < ERROR < CRITICAL
    the default value is logging.INFO
    when - how to split the log file by time interval
    'S' : Seconds
    'M' : Minutes
    'H' : Hours
    'D' : Days
    'W' : Week day
    default value: 'D'
    format - format of the log
    default format:
    %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
    INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
    backup - how many backup file to keep
    default value: 7
    Raises:
    OSError: fail to create log directories
    IOError: fail to open log file
    """
    formatter = logging.Formatter(format, datefmt)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(level)
    dir = os.path.dirname(log_path)
    if not os.path.isdir(dir) and not os.path.exists(dir):
        os.makedirs(dir)
    handler = logging.handlers.TimedRotatingFileHandler(log_path + ".log",
                                                        when=when,
                                                        backupCount=backup)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    handler = logging.handlers.TimedRotatingFileHandler(log_path + ".log.wf",
                                                        when=when,
                                                        backupCount=backup)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console_hdl = logging.StreamHandler(stream=sys.stdout)
    console_hdl.setFormatter(formatter)
    logger.addHandler(console_hdl)
