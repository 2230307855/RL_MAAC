import imp
import os.path as osp

# 它接受一个模块名称作为输入，并使用imp.load_source函数从与脚本相同目录中的文件中加载模块
def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)
