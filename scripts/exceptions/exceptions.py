import sys
from scripts.logging.logger import logging

def get_msg_detail(error_msg, error_detail: sys):
    '''This function returns the file name, error line, and error message.'''
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename 
        error_line = exc_tb.tb_lineno 
        error_message = f"Error occurred in script: [{file_name}] at line number: [{error_line}] with error message: [{error_msg}]"
    else:
        error_message = f"Error occurred with message: [{error_msg}], but traceback is not available."
    return error_message

def log_exception(error_msg, error_detail: sys):
    '''This function logs the exception details.'''
    msg_detail = get_msg_detail(error_msg, error_detail)
    logging.error(msg_detail)

class DIMException(Exception):
    def __init__(self, error_msg, error_detail: sys):
        super().__init__(error_msg)
        self.error_msg = get_msg_detail(error_msg, error_detail)
        log_exception(self.error_msg, error_detail)

    def __str__(self):
        return self.error_msg
    
# Example usage:
# if __name__ == "__main__":
#     try:
#         logging.info("Testing DIMException")
#         x = 1 / 0  # This will cause ZeroDivisionError
#     except Exception as e:
#         raise DIMException(str(e), sys)