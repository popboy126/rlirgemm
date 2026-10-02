# -*- coding: utf-8 -*-
# Author        : huche
# Create Time   : 2025/3/31 11:25
# Contact       : hucheng.lew@qq.com
# File          : gemm_agent.py
# IDE           : PyCharm
# Description   : gemm代理代码
import logging
import os
import re
import time
from typing import List, Optional, Tuple

from fabric import Connection
from invoke import CommandTimedOut, Responder, Result


class SSHAgent:
    "矩阵乘法环境的agent基类"

    def __init__(
        self,
        user,
        ip,
        port,
        password,
        gemm_agent_name: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        记录一些权限信息
        """
        if logger is None:
            self._logger = logging.getLogger(gemm_agent_name)
        else:
            self._logger = logger
        self.user = user
        self.ip = ip
        self.port = port
        self.password = password
        self.connection = self.connect()

        self._WATCHERS = [
            Responder(
                pattern=r"Are you sure you want to continue connecting.*",
                response="yes\n",
            ),
            Responder(
                pattern="{}@.*'s password:.*".format(self.user),
                response="{}\n".format(self.password),
            ),
            Responder(
                pattern=r"Enter file in which to save the key.*",
                response="\n",
            ),
            Responder(
                pattern=r"Overwrite.*",
                response="y\n",
            ),
        ]
        self._PARSE_GEMM_PERFORMANCE_REGEX = re.compile(
            r"^\[output\] benchmark: (\S+) GFlops"
        )

    def connect(self, timeout: int = 180) -> Connection:
        """
        连接到环境服务器
        """
        return Connection(
            host=self.ip,
            user=self.user,
            port=self.port,
            connect_timeout=timeout,
            connect_kwargs={
                "password": self.password,
            },
        )

    def run(self, command, **kwargs) -> Optional[Result]:
        """
        运行命令
        :param command: 命令
        :param kwargs: 其他参数
        :return: 返回结果
        """
        retry_times = 3
        timeout = 10 * 60
        while retry_times > 0:
            try:
                self.connection.open()
                result = self.connection.run(command, timeout=timeout, **kwargs)
                return result
            except (ConnectionError, ConnectionAbortedError, ConnectionResetError) as e:
                time.sleep(180)
                self._logger.warning(
                    "ssh connection is not ok due to {}. retry times: {}".format(
                        e, retry_times
                    )
                )
                retry_times = retry_times - 1
                if retry_times == 0:
                    raise e
            except CommandTimedOut as e:
                timeout += 5 * 60
                self._logger.warning(
                    "command run timeout due to {}. retry times: {}, new timeout: {}".format(
                        e, retry_times, timeout
                    )
                )
                retry_times = retry_times - 1
                if retry_times == 0:
                    raise e
            except Exception as e:
                self._logger.error("run command failed. error: {}".format(e))
                raise e

        return None

    def run_gemm(
        self,
        work_dir: str,
        bin_path: str,
        storage_layout: int,
        loop_order: int,
        packing_type: int,
        element_precision: int,
        m_thread: int,
        k_thread: int,
        n_thread: int,
        mc: int,
        kc: int,
        nc: int,
        gemm_mat_a_prfm_stride: int,
        gemm_mat_b_prfm_stride: int,
        gemm_mat_c_prfm_stride: int,
        add_mat_c_prfm_stride: int,
        add_partial_sum_prfm_stride: int,
        trail_repeat_times: int,
        loop_times: int,
        M: int,
        N: int,
        K: int,
        alpha: float,
        beta: float,
        bind_cores: str,
    ) -> Tuple[int, List[float]]:
        """
        运行GEMM计算
        """
        try:
            # 构建mm.conf配置文件
            ret = self.run(
                f'echo "{element_precision} {storage_layout} {loop_order} {packing_type} {m_thread} {k_thread} {n_thread} {mc} {kc} {nc} {gemm_mat_a_prfm_stride} {gemm_mat_b_prfm_stride} {gemm_mat_c_prfm_stride} {add_mat_c_prfm_stride} {add_partial_sum_prfm_stride} 1 {loop_times} {M} {K} {N} {alpha} {beta}" > {work_dir}/mm.conf',
                pty=False,
                watchers=self._WATCHERS,
            )
            if ret.failed:
                raise Exception(
                    "mm.conf file create failed. return_code: {}, stderr: {}, stdout: {}".format(
                        ret.return_code, ret.stderr, ret.stdout
                    )
                )

            results = list()
            for trial_id in range(trail_repeat_times):
                # 运行GEMM计算
                ret_run = self.run(
                    f"taskset -c {bind_cores} {bin_path} {work_dir}/mm.conf",
                    pty=False,
                    watchers=self._WATCHERS,
                )
                if ret_run.failed:
                    raise Exception(
                        "run gemm test {} failed. return_code: {}, stderr: {}, stdout: {}".format(
                            trial_id,
                            ret_run.return_code,
                            ret_run.stderr,
                            ret_run.stdout,
                        )
                    )

                self._logger.debug(
                    "run gemm test {} succeed. return_code: {}, stderr: {}, stdout: {}".format(
                        trial_id, ret_run.return_code, ret_run.stderr, ret_run.stdout
                    )
                )
                # 解析结果
                out = ret_run.stdout
                lines = out.split("\n")
                for line in lines:
                    search_result = self._PARSE_GEMM_PERFORMANCE_REGEX.search(line)
                    if search_result:
                        results.append(float(search_result.group(1)))
                        break
        except Exception as e:
            self._logger.error("run gemm failed. error: {}".format(e))
            return -1, []

        if not results:
            results = [0.0 for _ in range(trail_repeat_times)]

        return 0, results
