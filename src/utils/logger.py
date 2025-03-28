import logging

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler('training.log')
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger

def log_training_info(logger, epoch, loss):
    logger.info(f'Epoch: {epoch}, Loss: {loss}')