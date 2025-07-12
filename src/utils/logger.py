"""
日志工具
设置和管理日志记录
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: str,
                log_file: Optional[str] = None,
                level: int = logging.INFO,
                format_str: Optional[str] = None) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径（可选）
        level: 日志级别
        format_str: 日志格式字符串
        
    Returns:
        配置好的日志器
    """
    # 默认格式
    if format_str is None:
        format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 如果已经有处理器，先清除
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建格式器
    formatter = logging.Formatter(format_str)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file is not None:
        # 确保目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取已存在的日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        日志器实例
    """
    return logging.getLogger(name) 