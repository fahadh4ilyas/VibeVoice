import os, logging, uvicorn
from app.config import config

ROOT_DIR = os.path.dirname(__file__)

os.makedirs(os.path.join(ROOT_DIR, 'logs'), exist_ok=True)

if __name__ == '__main__':

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
                'formatter': 'all_format',
                "stream": "ext://sys.stdout"
            },
            'error': {
                'class': "logging.handlers.TimedRotatingFileHandler",
                'formatter': 'all_format',
                'filename': os.path.join(ROOT_DIR, 'logs', 'debug.log'),
                'when': 'midnight',
                'backupCount': 30, # Keep 30 days of logs
            },
            'access': {
                'class': "logging.handlers.TimedRotatingFileHandler",
                'formatter': 'all_format',
                'filename': os.path.join(ROOT_DIR, 'logs', 'access.log'),
                'when': 'midnight',
                'backupCount': 30, # Keep 30 days of logs
            }
        },
        'loggers': {
            'gunicorn.access': {
                'handlers': ['console', 'access'],
                'level': 'INFO',
                'propagate': False
            },
            'uvicorn.access': {
                'handlers': ['console', 'access'],
                'level': 'INFO',
                'propagate': False
            },
            'gunicorn.error': {
                'handlers': ['console', 'error'],
                'level': 'INFO',
                'propagate': False
            },
            'uvicorn.error': {
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

    uvicorn.run(
        'app.main:app',
        host=config.api_host,
        port=config.api_port,
        workers=config.worker_num,
        forwarded_allow_ips=config.forwarded_allow_ips,
        log_config=logconfig_dict
    )