#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gate.io量化交易系统 - Web界面

使用Streamlit构建的交易系统Web界面，提供策略配置、市场数据查看和交易监控功能。
"""
import os
import sys
import streamlit as st

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入页面模块
from ui.trading import render_trading_page
from ui.market_data import render_market_data_page
from ui.account import render_account_page

# 设置页面配置
st.set_page_config(
    page_title="Gate.io量化交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 定义页面
PAGES = {
    "交易中心": render_trading_page,
    "市场数据": render_market_data_page,
    "账户资产": render_account_page
}

# 侧边栏 - 页面导航
st.sidebar.title("Gate.io量化交易系统")

# 创建一个简单的logo
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px;">
    <h1 style="color: #1E88E5; font-size: 24px;">📈 BTC Trading</h1>
</div>
""", unsafe_allow_html=True)

# 选择页面
selection = st.sidebar.radio("导航", list(PAGES.keys()))

# 渲染选中的页面
page = PAGES[selection]
page()

# 添加页脚
st.sidebar.markdown("---")
st.sidebar.info(
    """
    Gate.io量化交易系统 | 版本 1.0.0

    © 2023 All Rights Reserved
    """
)
