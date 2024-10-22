"""
Utils package initialization.
"""
from .logger import logger
from .scheduler import init_scheduler

__all__ = ['logger', 'init_scheduler']