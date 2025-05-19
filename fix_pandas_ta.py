#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 pandas-ta 包中的 squeeze_pro.py 文件导入错误
"""
import os
import sys

def fix_squeeze_pro():
    """
    修复 squeeze_pro.py 文件中的导入错误
    将 'from numpy import NaN as npNaN' 改为 'import numpy as np; npNaN = np.nan'
    """
    # 查找 pandas_ta 包的路径
    import pandas_ta
    pandas_ta_path = os.path.dirname(pandas_ta.__file__)
    squeeze_pro_path = os.path.join(pandas_ta_path, 'momentum', 'squeeze_pro.py')
    
    print(f"正在修复文件: {squeeze_pro_path}")
    
    # 读取文件内容
    with open(squeeze_pro_path, 'r') as f:
        content = f.read()
    
    # 替换导入语句
    if 'from numpy import NaN as npNaN' in content:
        content = content.replace(
            'from numpy import NaN as npNaN',
            'import numpy as np\nnpNaN = np.nan'
        )
        print("已替换导入语句")
        
        # 写回文件
        with open(squeeze_pro_path, 'w') as f:
            f.write(content)
        print("文件已修复")
    else:
        print("未找到需要替换的导入语句，文件可能已经被修改或不存在此问题")

if __name__ == "__main__":
    fix_squeeze_pro()
    print("修复完成，请重新运行您的程序")
