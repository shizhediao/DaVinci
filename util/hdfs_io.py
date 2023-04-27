# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from typing import IO, Any, List

import shutil
import subprocess
from contextlib import contextmanager
import os
import glob
import threading

HADOOP_BIN = 'HADOOP_ROOT_LOGGER=ERROR,console /opt/tiger/yarn_deploy/hadoop/bin/hdfs'

__all__ = ['hlist_files', 'hopen', 'hexists', 'hmkdir', 'hglob', 'hisdir', 'hcountline']


@contextmanager  # type: ignore
def hopen(hdfs_path: str, mode: str = "r") -> IO[Any]:
    """
        打开一个 hdfs 文件, 用 contextmanager.

        Args:
            hfdfs_path (str): hdfs文件路径
            mode (str): 打开模式，支持 ["r", "w", "wa"]
    """
    pipe = None
    if mode.startswith("r"):
        pipe = subprocess.Popen(
            "{} dfs -text {}".format(HADOOP_BIN, hdfs_path), shell=True, stdout=subprocess.PIPE)
        yield pipe.stdout
        pipe.stdout.close()  # type: ignore
        pipe.wait()
        return
    if mode == "wa" or mode == "a":
        pipe = subprocess.Popen(
            "{} dfs -appendToFile - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    if mode.startswith("w"):
        pipe = subprocess.Popen(
            "{} dfs -put -f - {}".format(HADOOP_BIN, hdfs_path), shell=True, stdin=subprocess.PIPE)
        yield pipe.stdin
        pipe.stdin.close()  # type: ignore
        pipe.wait()
        return
    raise RuntimeError("unsupported io mode: {}".format(mode))


def hlist_files(folders: List[str]) -> List[str]:
    """
        罗列一些 hdfs 路径下的文件。

        Args:
            folders (List): hdfs文件路径的list
        Returns:
            一个list of hdfs 路径
    """
    files = []
    for folder in folders:
        if folder.startswith('hdfs'):
            pipe = subprocess.Popen("{} dfs -ls {}".format(HADOOP_BIN, folder), shell=True,
                                    stdout=subprocess.PIPE)
            # output, _ = pipe.communicate()
            for line in pipe.stdout:  # type: ignore
                line = line.strip()
                # drwxr-xr-x   - user group  4 file
                if len(line.split()) < 5:
                    continue
                files.append(line.split()[-1].decode("utf8"))
            pipe.stdout.close()  # type: ignore
            pipe.wait()
        else:
            if os.path.isdir(folder):
                files.extend([os.path.join(folder, d) for d in os.listdir(folder)])
            elif os.path.isfile(folder):
                files.append(folder)
            else:
                print('Path {} is invalid'.format(folder))
                sys.stdout.flush()

    return files


def hexists(file_path: str) -> bool:
    """ hdfs capable to check whether a file_path is exists """
    if file_path.startswith('hdfs'):
        return os.system("{} dfs -test -e {}".format(HADOOP_BIN, file_path)) == 0
    return os.path.exists(file_path)


def hisdir(file_path: str) -> bool:
    """ hdfs capable to check whether a file_path is a dir """
    if file_path.startswith('hdfs'):
        flag1 = os.system("{} dfs -test -e {}".format(HADOOP_BIN, file_path))  # 0:路径存在
        flag2 = os.system("{} dfs -test -f {}".format(HADOOP_BIN, file_path))  # 0:是文件 1:不是文件
        flag = ((flag1 == 0) and (flag2 == 1))
        return flag
    return os.path.isdir(file_path)


def hmkdir(file_path: str) -> bool:
    """ hdfs mkdir """
    if file_path.startswith('hdfs'):
        os.system("{} dfs -mkdir -p {}".format(HADOOP_BIN, file_path))
    else:
        os.mkdir(file_path)
    return True


def hcopy(from_path: str, to_path: str) -> bool:
    """ hdfs copy """
    if to_path.startswith("hdfs"):
        if from_path.startswith("hdfs"):
            os.system("{} dfs -cp -f {} {}".format(HADOOP_BIN, from_path, to_path))
        else:
            os.system("{} dfs -copyFromLocal -f {} {}".format(HADOOP_BIN, from_path, to_path))
    else:
        if from_path.startswith("hdfs"):
            os.system("{} dfs -text {} > {}".format(HADOOP_BIN, from_path, to_path))
        else:
            shutil.copy(from_path, to_path)
    return True


def hglob(search_path, sort_by_time=False):
    """ hdfs glob """
    if search_path.startswith("hdfs"):
        if sort_by_time:
            hdfs_command = HADOOP_BIN + ' dfs -ls %s | sort -k6,7' % search_path
        else:
            hdfs_command = HADOOP_BIN + ' dfs -ls %s' % search_path
        path_list = []
        files = os.popen(hdfs_command).read()
        files = files.split("\n")
        for file in files:
            if 'hdfs' in file:
                startindex = file.index('hdfs')
                path_list.append(file[startindex:])
        return path_list
    else:
        files = glob.glob(search_path)
        if sort_by_time:
            files = sorted(files, key=lambda x: os.path.getmtime(x))
    return files


def htext_list(files, target_folder):
    for fn in files:
        name = fn.split('/')[-1]
        hdfs_command = HADOOP_BIN + ' dfs -text %s > %s/%s' % (fn, target_folder, name)
        os.system(hdfs_command)


def hmget(files, target_folder, num_thread=16):
    """ 将整个hdfs 文件夹 get下来，但是不是简单的get，因为一些hdfs文件是压缩的，需要解压"""
    part = len(files) // num_thread
    thread_list = []
    for i in range(num_thread):
        start = part * i
        if i == num_thread - 1:
            end = len(files)
        else:
            end = start + part
        t = threading.Thread(target=htext_list, kwargs={
                             'files': files[start:end], 'target_folder': target_folder})
        thread_list.append(t)

    for t in thread_list:
        t.setDaemon(True)
        t.start()

    for t in thread_list:
        t.join()


def hcountline(path):
    '''
    count line in file
    '''
    count = 0
    if path.startswith('hdfs'):
        with hopen(path, 'r') as f:
            for line in f:
                count += 1
    else:
        with open(path, 'r') as f:
            for line in f:
                count += 1
    return count
