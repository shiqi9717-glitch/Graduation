"""
项目配置设置
使用 pydantic 进行环境变量验证和类型安全配置
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Settings:
    """项目配置类"""
    
    # API 配置
    GLOBAL_API_KEY: str = os.getenv("GLOBAL_API_KEY", "")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # 请求参数
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
    
    # 温度参数
    TEMPERATURE_FOR_GENERATION: float = 0.8  # 生成回答时的温度
    TEMPERATURE_FOR_JUDGMENT: float = 0.0    # 裁判打分时的温度
    
    # 数据路径
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    RAW_DATA_PATH: Path = PROJECT_ROOT / os.getenv("RAW_DATA_PATH", "data/raw/sycophancy_questions.csv")
    PROCESSED_DATA_PATH: Path = PROJECT_ROOT / os.getenv("PROCESSED_DATA_PATH", "data/processed/sycophancy_ready_to_test.csv")
    RESULTS_PATH: Path = PROJECT_ROOT / os.getenv("RESULTS_PATH", "data/results/judgment_results.csv")
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[Path] = PROJECT_ROOT / os.getenv("LOG_FILE", "logs/alignment_tax.log") if os.getenv("LOG_FILE") else None
    
    # 实验参数
    BATCH_SIZE: int = 5  # 异步批处理大小
    MAX_CONCURRENT_REQUESTS: int = 10  # 最大并发请求数
    
    def validate(self) -> bool:
        """验证配置是否有效"""
        if not self.GLOBAL_API_KEY:
            raise ValueError("GLOBAL_API_KEY 未设置，请在 .env 文件中配置")
        
        # 确保数据目录存在
        self.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        if self.LOG_FILE:
            self.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        return True
    
    @property
    def api_headers(self) -> dict:
        """获取 API 请求头"""
        return {
            "Authorization": f"Bearer {self.GLOBAL_API_KEY}",
            "Content-Type": "application/json"
        }


# 全局配置实例
settings = Settings()
