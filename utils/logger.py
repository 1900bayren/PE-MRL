import logging
import os
import sys
import os.path as osp
from datetime import datetime

def setup_logger(name, save_dir, if_train):
    """
       配置并创建一个日志记录器(logger)，支持同时输出日志到控制台和文件

       参数:
           name (str): 日志记录器的名称，用于标识不同的日志器实例
           save_dir (str): 日志文件保存的目录路径，如果为None则不保存到文件
           if_train (bool): 标识当前是训练阶段还是测试阶段，用于区分日志文件名

       返回:
           logging.Logger: 配置好的日志记录器实例
       """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # 设置日志器的全局级别为DEBUG（最低级别，确保所有级别的日志都能被处理）

    ch = logging.StreamHandler(stream=sys.stdout)  # 创建控制台日志处理器(StreamHandler)，日志将输出到标准输出(stdout)
    ch.setLevel(logging.DEBUG) # 设置控制台日志的级别为DEBUG（处理所有级别的日志
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s") # 定义日志格式：包含时间、日志器名称、日志级别和日志消息
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        # 生成时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if if_train:
            filename = f"train_log_{timestamp}.log"
        else:
            filename = f"test_log_{timestamp}.log"

        fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
