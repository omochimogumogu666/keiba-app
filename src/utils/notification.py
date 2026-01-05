"""
Notification utilities for model retraining and alerts.

Supports:
- Email notifications via SMTP
- Slack notifications (optional)
- Custom notification handlers
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from datetime import datetime

from src.utils.logger import get_app_logger

logger = get_app_logger(__name__)


class EmailNotifier:
    """Send email notifications for retraining events."""

    def __init__(self, config: Dict):
        """
        Initialize email notifier.

        Args:
            config: Email configuration dict with keys:
                - smtp_server: SMTP server address
                - smtp_port: SMTP port (default: 587)
                - smtp_user: SMTP username
                - smtp_password: SMTP password
                - from_email: Sender email address (optional, defaults to smtp_user)
        """
        self.config = config
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.smtp_user = config.get('smtp_user')
        self.smtp_password = config.get('smtp_password')
        self.from_email = config.get('from_email', self.smtp_user)

    def send_retraining_notification(
        self,
        recipients: List[str],
        results: List[Dict],
        status: str = 'success'
    ):
        """
        Send notification about retraining completion.

        Args:
            recipients: List of recipient email addresses
            results: List of model training results
            status: Status of retraining ('success' or 'failure')
        """
        if not recipients:
            logger.warning("No recipients configured for email notification")
            return

        subject = f"[競馬予想AI] モデル再訓練 {status.upper()}"

        # Build email body
        if status == 'success':
            body = self._build_success_email(results)
        else:
            body = self._build_failure_email(results)

        try:
            self._send_email(recipients, subject, body)
            logger.info(f"Notification email sent to {len(recipients)} recipients")
        except Exception as e:
            logger.error(f"Failed to send notification email: {e}", exc_info=True)

    def _build_success_email(self, results: List[Dict]) -> str:
        """Build email body for successful retraining."""
        body = "モデルの再訓練が完了しました。\n\n"
        body += "=" * 60 + "\n"
        body += "再訓練結果サマリー\n"
        body += "=" * 60 + "\n\n"

        for result in results:
            body += f"モデル: {result['model_type'].upper()}\n"
            body += f"  ファイル名: {result['filename']}\n"
            body += f"  訓練サンプル数: {result['train_samples']:,}\n"
            body += f"  テストサンプル数: {result['test_samples']:,}\n"
            body += f"  訓練時間: {result['training_time_seconds']}秒\n"
            body += f"  テストRMSE: {result['test_metrics']['rmse']:.4f}\n"
            body += f"  テストR2: {result['test_metrics']['r2_score']:.4f}\n"

            comparison = result.get('comparison', {})
            if comparison.get('has_previous'):
                improvement = comparison['improvement'] * 100
                body += f"  前回モデルとの比較: {improvement:+.2f}%\n"
                body += f"  デプロイ: {'はい' if comparison['should_deploy'] else 'いいえ'}\n"
            else:
                body += f"  デプロイ: はい (初回モデル)\n"

            body += "\n"

        body += "=" * 60 + "\n"
        body += f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        return body

    def _build_failure_email(self, error_info: Dict) -> str:
        """Build email body for failed retraining."""
        body = "モデルの再訓練が失敗しました。\n\n"
        body += "=" * 60 + "\n"
        body += "エラー情報\n"
        body += "=" * 60 + "\n\n"

        if isinstance(error_info, dict):
            body += f"エラー: {error_info.get('error', 'Unknown error')}\n"
            body += f"タイムスタンプ: {error_info.get('timestamp', 'Unknown')}\n"
        else:
            body += f"エラー: {str(error_info)}\n"

        body += "\n" + "=" * 60 + "\n"
        body += f"失敗時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        body += "\nログを確認してください。\n"

        return body

    def _send_email(self, recipients: List[str], subject: str, body: str):
        """
        Send email via SMTP.

        Args:
            recipients: List of recipient email addresses
            subject: Email subject
            body: Email body text
        """
        # Create message
        msg = MIMEMultipart()
        msg['From'] = self.from_email
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = subject

        # Add body
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # Send email
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)


class SlackNotifier:
    """Send Slack notifications for retraining events."""

    def __init__(self, webhook_url: str):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url

    def send_retraining_notification(
        self,
        results: List[Dict],
        status: str = 'success'
    ):
        """
        Send notification about retraining completion.

        Args:
            results: List of model training results
            status: Status of retraining ('success' or 'failure')
        """
        try:
            import requests

            if status == 'success':
                message = self._build_success_message(results)
                color = 'good'
            else:
                message = self._build_failure_message(results)
                color = 'danger'

            payload = {
                'attachments': [
                    {
                        'color': color,
                        'title': f'モデル再訓練 {status.upper()}',
                        'text': message,
                        'footer': '競馬予想AI',
                        'ts': int(datetime.now().timestamp())
                    }
                ]
            }

            response = requests.post(self.webhook_url, json=payload)
            response.raise_for_status()

            logger.info("Slack notification sent successfully")

        except ImportError:
            logger.warning("requests library not installed, cannot send Slack notification")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}", exc_info=True)

    def _build_success_message(self, results: List[Dict]) -> str:
        """Build Slack message for successful retraining."""
        lines = []

        for result in results:
            model_type = result['model_type'].upper()
            rmse = result['test_metrics']['rmse']
            r2 = result['test_metrics']['r2_score']

            comparison = result.get('comparison', {})
            if comparison.get('has_previous'):
                improvement = comparison['improvement'] * 100
                deployed = '✅' if comparison['should_deploy'] else '⏸️'
                lines.append(
                    f"{deployed} *{model_type}*: RMSE={rmse:.4f}, R2={r2:.4f} "
                    f"({improvement:+.2f}% vs 前回)"
                )
            else:
                lines.append(
                    f"✅ *{model_type}*: RMSE={rmse:.4f}, R2={r2:.4f} (初回モデル)"
                )

        return '\n'.join(lines)

    def _build_failure_message(self, error_info: Dict) -> str:
        """Build Slack message for failed retraining."""
        if isinstance(error_info, dict):
            return f"❌ エラー: {error_info.get('error', 'Unknown error')}"
        return f"❌ エラー: {str(error_info)}"


def send_notification(config: Dict, results: List[Dict], status: str = 'success'):
    """
    Send notifications via configured channels.

    Args:
        config: Notification configuration
        results: List of model training results
        status: Status of retraining ('success' or 'failure')
    """
    if not config.get('enabled', False):
        logger.debug("Notifications are disabled")
        return

    # Email notification
    if config.get('email_recipients'):
        try:
            notifier = EmailNotifier(config)
            notifier.send_retraining_notification(
                config['email_recipients'],
                results,
                status
            )
        except Exception as e:
            logger.error(f"Email notification failed: {e}")

    # Slack notification
    if config.get('slack_webhook_url'):
        try:
            notifier = SlackNotifier(config['slack_webhook_url'])
            notifier.send_retraining_notification(results, status)
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
