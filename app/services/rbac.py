from __future__ import annotations

from pydantic import BaseModel, Field


class RolePolicy(BaseModel):
    name: str
    description: str
    permissions: set[str] = Field(default_factory=set)


COMMON_ANALYSIS_PERMISSIONS = {
    "view_summary",
    "view_signals",
    "view_charts",
    "view_news",
    "view_backtest",
    "view_ml_models",
    "view_correlation",
    "view_data_sources",
    "view_risks",
    "view_strategies",
}

WATCHLIST_PERMISSIONS = {
    "view_watchlist",
    "create_watchlist",
    "edit_watchlist",
    "delete_watchlist",
    "add_instruments",
    "edit_instruments",
    "view_instruments",
}

PORTFOLIO_PERMISSIONS = {
    "view_portfolio",
    "create_portfolio",
    "edit_portfolio",
    "manage_portfolio",
    "view_portfolio_performance",
}

ALERT_PERMISSIONS = {
    "view_alerts",
    "create_alerts",
    "edit_alerts",
    "delete_alerts",
    "manage_alerts",
}

ACCOUNT_PERMISSIONS = {
    "view_account",
    "update_account",
    "delete_account",
    "view_account_profile",
    "update_account_profile",
    "view_email",
    "update_email",
    "view_security",
    "update_password",
    "manage_sessions",
    "view_preferences",
    "update_preferences",
}

SUBSCRIPTION_PERMISSIONS = {
    "view_subscription",
    "update_subscription",
    "cancel_subscription",
    "view_invoices",
    "download_invoices",
    "manage_billing",
}

PREMIUM_ANALYSIS_PERMISSIONS = {
    "add_buy_indicators",
    "add_sell_indicators",
    "edit_chart_indicators",
    "view_realtime_analysis",
    "sync_workspace",
    "sync_instrument",
}

ADMIN_PERMISSIONS = {
    "manage_users",
    "manage_roles",
    "manage_access",
    "view_users",
    "edit_users",
    "archive_users",
    "manage_password_policy",
    "manage_identity_providers",
    "manage_subscriptions",
}

ROLE_POLICIES: dict[str, RolePolicy] = {
    "guest": RolePolicy(
        name="guest",
        description="Temporary browser/session user with analysis-only access and no persisted portfolio.",
        permissions=COMMON_ANALYSIS_PERMISSIONS | {"view_watchlist"},
    ),
    "user": RolePolicy(
        name="user",
        description="Signed-in user with watchlist, portfolio, alert, and instrument-analysis access.",
        permissions=COMMON_ANALYSIS_PERMISSIONS
        | WATCHLIST_PERMISSIONS
        | PORTFOLIO_PERMISSIONS
        | ALERT_PERMISSIONS
        | ACCOUNT_PERMISSIONS
        | SUBSCRIPTION_PERMISSIONS
        | PREMIUM_ANALYSIS_PERMISSIONS,
    ),
    "admin": RolePolicy(
        name="admin",
        description="Workspace administrator with user, access, and configuration permissions.",
        permissions=COMMON_ANALYSIS_PERMISSIONS
        | WATCHLIST_PERMISSIONS
        | PORTFOLIO_PERMISSIONS
        | ALERT_PERMISSIONS
        | ACCOUNT_PERMISSIONS
        | SUBSCRIPTION_PERMISSIONS
        | PREMIUM_ANALYSIS_PERMISSIONS
        | ADMIN_PERMISSIONS,
    ),
    "superadmin": RolePolicy(
        name="superadmin",
        description="Unrestricted platform administrator.",
        permissions=COMMON_ANALYSIS_PERMISSIONS
        | WATCHLIST_PERMISSIONS
        | PORTFOLIO_PERMISSIONS
        | ALERT_PERMISSIONS
        | ACCOUNT_PERMISSIONS
        | SUBSCRIPTION_PERMISSIONS
        | PREMIUM_ANALYSIS_PERMISSIONS
        | ADMIN_PERMISSIONS
        | {"view_all_users", "view_all_subscriptions", "restore_archives"},
    ),
}

ROLE_DESCRIPTIONS = {name: policy.description for name, policy in ROLE_POLICIES.items()}
ROLE_PERMISSIONS = {name: sorted(policy.permissions) for name, policy in ROLE_POLICIES.items()}


def normalize_role(role: str | None) -> str:
    candidate = str(role or "guest").strip().lower()
    if candidate == "auth_user":
        candidate = "user"
    return candidate if candidate in ROLE_POLICIES else "guest"


def permissions_for_role(role: str | None) -> set[str]:
    return set(ROLE_POLICIES[normalize_role(role)].permissions)


def role_has_permission(role: str | None, permission: str) -> bool:
    return permission in permissions_for_role(role)
