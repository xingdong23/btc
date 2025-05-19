"""
通知模块

提供邮件、短信等通知功能。
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict, Any

from config.config import config
from utils.logger import log


class NotificationManager:
    """通知管理器类"""
    
    def __init__(self):
        """初始化通知管理器"""
        self.enable_email = config.get('notifications', 'enable_email', 'false').lower() == 'true'
        self.email_to = config.get('notifications', 'email', '')
        self.smtp_server = config.get('notifications', 'smtp_server', '')
        self.smtp_port = int(config.get('notifications', 'smtp_port', '587'))
        self.smtp_user = config.get('notifications', 'smtp_user', '')
        self.smtp_password = config.get('notifications', 'smtp_password', '')
    
    def send_email(self, subject: str, message: str, to_email: Optional[str] = None) -> bool:
        """
        发送邮件通知
        
        Args:
            subject: 邮件主题
            message: 邮件内容
            to_email: 收件人邮箱，如果为None则使用配置中的邮箱
            
        Returns:
            bool: 是否发送成功
        """
        if not self.enable_email:
            log.info("邮件通知功能未启用")
            return False
        
        if not self.smtp_server or not self.smtp_user or not self.smtp_password:
            log.error("邮件服务器配置不完整")
            return False
        
        to_email = to_email or self.email_to
        if not to_email:
            log.error("未指定收件人邮箱")
            return False
        
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.smtp_user
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # 添加邮件内容
            msg.attach(MIMEText(message, 'plain'))
            
            # 连接SMTP服务器并发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()  # 启用TLS加密
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            log.info(f"邮件通知已发送至 {to_email}")
            return True
            
        except Exception as e:
            log.error(f"发送邮件通知失败: {str(e)}")
            return False
    
    def send_trade_notification(self, trade: Dict[str, Any]) -> bool:
        """
        发送交易通知
        
        Args:
            trade: 交易信息
            
        Returns:
            bool: 是否发送成功
        """
        if not self.enable_email:
            return False
        
        # 构建通知内容
        subject = f"交易通知: {trade['side'].upper()} {trade['symbol']}"
        
        message = f"""
        交易详情:
        --------
        交易对: {trade['symbol']}
        操作: {trade['side'].upper()}
        价格: {trade['price']}
        数量: {trade['amount']}
        时间: {trade['datetime']}
        订单ID: {trade['id']}
        """
        
        return self.send_email(subject, message)
    
    def send_alert(self, title: str, content: str, level: str = 'info') -> bool:
        """
        发送警报通知
        
        Args:
            title: 警报标题
            content: 警报内容
            level: 警报级别，可选值: info, warning, error
            
        Returns:
            bool: 是否发送成功
        """
        if not self.enable_email:
            return False
        
        # 根据级别设置主题前缀
        prefix = {
            'info': '[信息]',
            'warning': '[警告]',
            'error': '[错误]'
        }.get(level.lower(), '[信息]')
        
        subject = f"{prefix} {title}"
        
        return self.send_email(subject, content)


# 全局通知管理器实例
notification_manager = NotificationManager()
