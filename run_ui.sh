#!/bin/bash

# 检查是否安装了streamlit
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit未安装，正在安装..."
    pip install streamlit
fi

# 检查是否安装了其他依赖
if [ -f "requirements.txt" ]; then
    echo "检查依赖..."
    pip install -r requirements.txt
fi

# 创建日志目录
mkdir -p logs

# 启动Streamlit应用
echo "启动交易系统界面..."
streamlit run app.py
