"""Who is using the dashboard and what they may do.

Roles
-----
- ``viewer``   can see forecasts, inventory health and recommendations.
- ``approver`` can additionally approve / reject transfer recommendations.

Modes (environment variables, see ``.env.example``)
---------------------------------------------------
- Sign-in required (``STOCKFLOW_REQUIRE_LOGIN=1``): Streamlit's built-in OIDC login
  (``st.login``) must be configured in ``.streamlit/secrets.toml``. A signed-in user is an
  approver only if their e-mail is listed in ``STOCKFLOW_APPROVERS`` (comma-separated).
- Demo (default when sign-in is not required): no identity provider. Everyone is a
  ``viewer``; setting ``STOCKFLOW_DEMO_APPROVALS=1`` lets a local demo user record decisions
  under a typed name, and the UI says so on every page. Never use demo mode for real data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    name: str
    email: str | None
    role: str  # "viewer" | "approver"
    authenticated: bool


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


def require_login() -> bool:
    return _flag("STOCKFLOW_REQUIRE_LOGIN")


def demo_approvals() -> bool:
    return _flag("STOCKFLOW_DEMO_APPROVALS")


def mask_financials() -> bool:
    """Hide dollar amounts (for screenshots / demos of real data)."""
    return _flag("STOCKFLOW_MASK_FINANCIALS")


def approvers() -> set[str]:
    return {e.strip().lower() for e in os.environ.get("STOCKFLOW_APPROVERS", "").split(",") if e.strip()}


def resolve(user_info: dict[str, object] | None) -> User:
    """Map the identity-provider claims (or their absence) to a ``User``.

    Args:
        user_info: ``st.user`` as a dict when signed in, else ``None``.
    """
    if user_info and user_info.get("is_logged_in"):
        email = str(user_info.get("email") or "").lower() or None
        role = "approver" if email and email in approvers() else "viewer"
        return User(str(user_info.get("name") or email or "signed-in user"), email, role, True)
    if require_login():
        return User("anonymous", None, "viewer", False)
    return User("demo user", None, "approver" if demo_approvals() else "viewer", False)
