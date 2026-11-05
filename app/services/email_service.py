import os
import smtplib
from email.message import EmailMessage

import requests

from dotenv import load_dotenv


load_dotenv()


def send_email(recipient: str, subject: str, body: str) -> bool:
    """Send transactional onboarding mail when SMTP is configured; never persist credentials."""
    host = os.getenv("SMTP_HOST", "")
    sender = os.getenv("SMTP_FROM", "")
    if not host or not sender:
        return False
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "")
    password = os.getenv("SMTP_PASSWORD", "")
    use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    try:
        with smtplib.SMTP(host, port, timeout=20) as client:
            if use_tls:
                client.starttls()
            if username:
                client.login(username, password)
            client.send_message(message)
        return True
    except (OSError, smtplib.SMTPException):
        # Account creation remains successful and the pending status enables a later resend.
        return False


def send_sms(recipient: str, body: str) -> bool:
    """Send an alert SMS through Twilio when SMS delivery is configured."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    sender = os.getenv("TWILIO_FROM_NUMBER", "")
    if not account_sid or not auth_token or not sender or not recipient:
        return False
    try:
        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json",
            auth=(account_sid, auth_token),
            data={"From": sender, "To": recipient, "Body": body},
            timeout=(5, 15),
        )
        response.raise_for_status()
        return True
    except (OSError, requests.RequestException):
        return False


def send_local_onboarding(recipient: str, name: str, reset_url: str) -> bool:
    return send_email(
        recipient,
        "Welcome to Stock Monitoring App",
        f"Hello {name},\n\nYour local Stock Monitoring App account is ready. "
        f"Activate your account using this one-time link:\n{reset_url}\n\n"
        "If you did not expect this invitation, contact your workspace administrator.",
    )


def send_sso_onboarding(recipient: str, name: str, provider_name: str) -> bool:
    return send_email(
        recipient,
        "BA Optimization SSO access",
        f"Hello {name},\n\nYour BA Optimization account was synchronized for tenant access. "
        f"Use your {provider_name} enterprise sign-in. No local password is required.",
    )
