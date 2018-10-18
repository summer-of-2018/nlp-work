
# config.py
import os
import gevent.monkey
gevent.monkey.patch_all()

import multiprocessing

# debug = False
loglevel = 'debug'
bind = "127.0.0.1:8888"
pidfile = "/root/logs/gunicorn.pid"
accesslog = "/root/logs/access.log"
errorlog = "/root/logs/debug.log"
daemon = True

# 启动的进程数
workers = multiprocessing.cpu_count()
worker_class = 'gevent'
x_forwarded_for_header = 'X-FORWARDED-FOR'
