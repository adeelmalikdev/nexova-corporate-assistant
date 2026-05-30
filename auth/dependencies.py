from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from auth.jwt_handler import _parse_allowed_domains, decode_token
from database import crud
from database.db import get_db


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def _as_utc(value: datetime | None) -> datetime | None:
	if value is None:
		return None
	if value.tzinfo is None:
		return value.replace(tzinfo=timezone.utc)
	return value.astimezone(timezone.utc)


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> dict:
	payload = decode_token(token)
	jti = payload.get("jti")
	if not isinstance(jti, str):
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid or expired token",
			headers={"WWW-Authenticate": "Bearer"},
		)
	if await crud.is_token_blacklisted(db, jti):
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Token has been revoked",
			headers={"WWW-Authenticate": "Bearer"},
		)

	username = payload.get("sub")
	if not isinstance(username, str) or not username:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid or expired token",
			headers={"WWW-Authenticate": "Bearer"},
		)

	employee = await crud.get_by_username(db, username)
	if employee is None:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid or expired token",
			headers={"WWW-Authenticate": "Bearer"},
		)

	if not employee.is_active:
		raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

	if not employee.is_verified:
		raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account pending verification")

	now = datetime.now(timezone.utc)
	locked_until = _as_utc(employee.locked_until)
	if locked_until and locked_until > now:
		seconds_remaining = max(int((locked_until - now).total_seconds()), 1)
		raise HTTPException(
			status_code=status.HTTP_423_LOCKED,
			detail=f"Account locked. Try again in {seconds_remaining} seconds.",
			headers={"Retry-After": str(seconds_remaining)},
		)

	role = employee.role
	allowed_domains = _parse_allowed_domains(role.allowed_domains) if role else []
	return {
		"employee": employee,
		"role": role.name if role else None,
		"role_name": role.name if role else None,
		"allowed_domains": allowed_domains,
		"force_password_change": not employee.password_changed,
		"department_id": employee.department_id,
	}


async def require_password_changed(current_user: dict = Depends(get_current_user)) -> dict:
	if current_user.get("force_password_change"):
		raise HTTPException(
			status_code=status.HTTP_403_FORBIDDEN,
			detail="Password change required before proceeding",
		)
	return current_user


async def require_admin(current_user: dict = Depends(require_password_changed)) -> dict:
	if current_user.get("role_name") != "admin":
		raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
	return current_user


async def require_hr_admin(current_user: dict = Depends(require_password_changed)) -> dict:
	if current_user.get("role_name") not in ["admin", "hr_admin"]:
		raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="HR admin access required")
	return current_user
