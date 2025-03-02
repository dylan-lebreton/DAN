# -*- coding: utf-8 -*-
# log.py
"""
File defining a logger to use in the project.
"""

import logging
import os


class FunctionFilter(logging.Filter):
    def filter(self, record):
        record.func_name = record.funcName
        record.class_name = record.module.split('.')[-1]
        return True

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(class_name)s.%(func_name)s - %(message)s')

# Create file handler and add formatter to it
log_file = os.path.join(os.getcwd(), "logs.txt")
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Create console handler and add formatter to it
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Add filter to logger
logger.addFilter(FunctionFilter())