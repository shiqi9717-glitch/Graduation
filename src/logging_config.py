"""
日志配置模块
提供统一的日志记录功能
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import settings


def setup_logger(
    name: str = "alignment_tax",
    log_file: Optional[Path] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    设置并返回配置好的日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，如果为 None 则只输出到控制台
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 全局日志记录器
logger = setup_logger(
    log_file=settings.LOG_FILE,
    level=settings.LOG_LEVEL
)


def get_progress_logger(total: int, desc: str = "处理进度") -> callable:
    """
    获取进度日志函数
    
    Args:
        total: 总任务数
        desc: 进度描述
        
    Returns:
        进度日志函数，调用时传入当前进度
    """
    from datetime import datetime
    
    start_time = datetime.now()
    
    def log_progress(current: int):
        """记录进度"""
        if current == 0:
            logger.info(f"开始{desc}，共{total}项任务")
            return
        
        elapsed = (datetime.now() - start_time).total_seconds()
        progress = current / total * 100
        
        if current == total:
            logger.info(f"{desc}完成！共处理{total}项，耗时{elapsed:.2f}秒")
        elif current % max(1, total // 10) == 0:  # 每10%记录一次
            remaining = elapsed / current * (total - current) if current > 0 else 0
            logger.info(
                f"{desc}: {current}/{total} ({progress:.1f}%) - "
                f"已耗时{elapsed:.1f}s，预计剩余{remaining:.1f}s"
            )
    
    return log_progress