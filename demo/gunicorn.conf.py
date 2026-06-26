import os, logging
from logging import handlers
from app.config import config as api_config

CURRENT_DIR = os.path.dirname(__file__)

os.makedirs(os.path.join(CURRENT_DIR, 'logs'), exist_ok=True)

# Worker
workers = api_config.worker_num
worker_class = "uvicorn.workers.UvicornWorker"
# Bind
bind = f'{api_config.api_host}:{api_config.api_port}'
# Forwarded IPs
forwarded_allow_ips = api_config.forwarded_allow_ips
# Logging
logconfig_dict = {
    'version': 1,
    'formatters': {
        'all_format': {
            'format': '[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S %z'
        }
    },
    'handlers': {
        'console': {
            'class': "logging.StreamHandler",
            'formatter': 'all_format'
        },
        'error': {
            'class': "logging.handlers.TimedRotatingFileHandler",
            'formatter': 'all_format',
            'filename': os.path.join(CURRENT_DIR, 'logs', 'debug.log'),
            'when': 'midnight',
            'backupCount': 30,  # Keep 30 days of logs
        },
        'access': {
            'class': "logging.handlers.TimedRotatingFileHandler",
            'formatter': 'all_format',
            'filename': os.path.join(CURRENT_DIR, 'logs', 'access.log'),
            'when': 'midnight',
            'backupCount': 30,  # Keep 30 days of logs
        }
    },
    'loggers': {
        'gunicorn.access': {
            'handlers': ['console', 'access'],
            'level': 'INFO',
            'propagate': False
        },
        'gunicorn.error': {
            'handlers': ['console', 'error'],
            'level': 'INFO',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['console', 'error'],
        'level': 'INFO'
    }
}
# Process Name
proc_name = 'gunicorn-vibevoice'
