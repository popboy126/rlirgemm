# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/7/4 17:37
# Contact       : hucheng.lew@qq.com
# File          : do_optuna_gemm_tuning_tasks.py
# IDE           : PyCharm
# Description   : 进行optuna性能搜索实验


import json
import logging
import logging.config
import os
import sys
from os import walk
from pathlib import Path
from typing import Any, Callable, List, TypeVar

T = TypeVar("T")


def import_api(
    script_dir: str | Path, module_name: str, func_name: str
) -> Callable[..., Any]:
    """
    导入各种API
    """
    scripts_dir = os.path.join(script_dir)
    # 添加项目目录到sys.path
    sys.path.append(scripts_dir)
    from importlib import import_module

    try:
        func_module = import_module(module_name)
        func_logic = getattr(func_module, func_name)
        return func_logic
    except ImportError as e:
        raise e
    except AttributeError as e:
        raise e


def find_optuna_gemm_tasks(
    project_dir: str,
    work_base_dir: str,
    task_conf_dir: str,
    task_conf_names: List[str] = None,
) -> None:
    """
    查找l1d缓存预取不同形状的测试
    :param task_conf_dir: 任务文件路径
    :param bin_root: 二进制文件根目录
    :param work_base_dir: 工作母目录
    :param task_conf_names: 支持可选任务名
    :return:
    """
    logger = logging.getLogger()
    # 查找任务文件
    task_json_list = list()
    if not task_conf_names:
        for root, dirs, files in walk(task_conf_dir):
            for i in files:
                if i.endswith("json"):
                    task_json_list.append(os.path.join(root, i))
    else:
        for i in task_conf_names:
            conf_path = os.path.join(task_conf_dir, i)
            if os.path.exists(conf_path) and os.path.isfile(conf_path):
                logger.info(f"config {conf_path} is checked~")
                task_json_list.append(os.path.join(task_conf_dir, i))
            else:
                raise FileNotFoundError(
                    f"config {conf_path} is not exists or not a file~"
                )

    logger.info(
        "find {} tasks. task list: {}".format(
            len(task_json_list), json.dumps(task_json_list)
        )
    )

    run_optuna_tuning = import_api(
        os.path.join(project_dir, "model", "optuna_search"),
        "do_optuna_search",
        "run",
    )

    for task_config in task_json_list:
        with open(task_config, "r") as ivl:
            task = json.load(ivl)
            # 创建工作目录
            work_dir = os.path.join(work_base_dir, task["name"])
            os.system("mkdir -p {}".format(work_dir))
            logger.info(
                "start to run task config: {}, work_dir: {}".format(
                    task_config, work_dir
                )
            )
            run_optuna_tuning(work_dir, task["name"], task["optuna_configs"])
        logger.info("task config {} is done~".format(task_config))


def init_logs(log_dir: str):
    """
    初始化日志
    :param log_dir:
    :return:
    """
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,  # 防止覆盖掉其他库已定义的 logger
        "formatters": {
            "standard": {
                "format": "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d]  %(message)s"
            },
        },
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
            "file": {
                "level": "INFO",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": os.path.join(log_dir, "optuna_tuning_tasks.log"),
                "encoding": "utf-8",
                "maxBytes": 1024 * 1024,
                "backupCount": 5,
                "formatter": "standard",
            },
        },
        "root": {
            "handlers": ["console", "file"],  # 这里定义了 Root Logger 使用的 handler
            "level": "DEBUG",
        },
    }
    # 应用配置
    logging.config.dictConfig(LOGGING_CONFIG)


if __name__ == "__main__":
    import sys

    project_dir = sys.argv[1]
    work_base_dir = sys.argv[2]
    task_config_dir = sys.argv[3]
    if len(sys.argv) > 4:  # 支持从命令行指定任务列表，避免再频繁的删了
        task_conf_names_str = sys.argv[4].strip()
        task_conf_names_list = task_conf_names_str.split(",")
    else:
        task_conf_names_list = None

    init_logs(work_base_dir)
    logger = logging.getLogger()
    logger.info("RL GEMM tasks start!!!!")

    find_optuna_gemm_tasks(
        project_dir, work_base_dir, task_config_dir, task_conf_names_list
    )
    logger.info("Optuna GEMM tuning tasks finished!!!!")
