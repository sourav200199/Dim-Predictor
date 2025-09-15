import os
import sys
import logging
import datetime as dt

LOG_FILE = f"{dt.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log"
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_PATH = os.path.join(os.getcwd(), LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename = LOG_PATH,
    level=logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(lineno)d - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)

