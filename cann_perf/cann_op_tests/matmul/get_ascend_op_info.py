# -*- coding: utf-8 -*-
"""用于导出OPINFO"""
import os
import sys
import argparse

def func(host_log_folder):
    """
    :param host_log_folder: where host_log_folder addr is.
    :return:
    """
    host_log_files = os.listdir(host_log_folder)
    result = {}

    for host_log in host_log_files:
        if not host_log.endswith('.log') or host_log.endswith('.out'):
            continue
        with open(os.path.join(host_log_folder, host_log), 'r')as f:
            host_log_lines = f.readlines()
            for line in host_log_lines:
                if line.startswith('[INFO] ASCENDCL') and "aclopCompile::aclOp" in line:
                    op_info = line.split('OpType: ')[1][:-2]
                    op_type = op_info.split(',')[0]
                    op_param = op_info[len(op_type) + 2:]
                    if op_type not in result.keys():
                        result[op_type] = [op_param]
                    else:
                        result[op_type].append(op_param)

    with open('ascend_op_info_summary.txt', 'w')as f:
        for k, v in result.items():
            v_set = set(v)
            for info in v_set:
                f.write(k + " " + info + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='trans the log')
    parser.add_argument('--host_log_folder', default="./",
                        help="input the dir name, trans the current dir with default")
    ags = parser.parse_args()
    func(ags.host_log_folder)
