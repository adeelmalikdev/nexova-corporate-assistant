from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user, oauth2_scheme, require_password_changed
from auth.jwt_handler import create_access_token, decode_token
from auth.password import check_password_history, hash_password, validate_password_strength, verify_password
from config import Settings
from database import crud
from database.db import get_db


settings = Settings()
router = APIRouter(tags=["Authentication"])


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


def _allowed_domains(raw_allowed_domains: str) -> list[str]:
    try:
        parsed = json.loads(raw_allowed_domains)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(domain) for domain in parsed]
    return []


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _serialize_employee(current_user: dict) -> dict:
    employee = current_user["employee"]
    role = employee.role
    department = employee.department
    return {
        "id": employee.id,
        "employee_id": employee.employee_id,
        "username": employee.username,
        "email": employee.email,
        "full_name": employee.full_name,
        "role": role.name if role else current_user.get("role_name"),
        "allowed_domains": current_user.get("allowed_domains", []),
        "department_id": employee.department_id,
        "department_name": department.display_name if department else None,
        "is_active": employee.is_active,
        "is_verified": employee.is_verified,
        "password_changed": employee.password_changed,
        "force_password_change": current_user.get("force_password_change", False),
        "last_login": employee.last_login,
        "locked_until": employee.locked_until,
        "temp_password_expires_at": employee.temp_password_expires_at,
    }


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)) -> dict:
    employee = await crud.get_by_username(db, form_data.username)
    if employee is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    now = datetime.now(timezone.utc)
    locked_until = _as_utc(employee.locked_until)
    if locked_until and locked_until > now:
        seconds_remaining = max(int((locked_until - now).total_seconds()), 1)
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Account locked. Try again in {seconds_remaining} seconds.",
            headers={"Retry-After": str(seconds_remaining)},
        )

    if not employee.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive")

    if not verify_password(form_data.password, employee.password_hash):
        await crud.record_failed_login(db, employee.id)
        updated_employee = await crud.get_by_id(db, employee.id)
        if updated_employee and updated_employee.failed_login_attempts >= settings.MAX_LOGIN_ATTEMPTS:
            lock_seconds = settings.LOCKOUT_MINUTES * 60
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail=f"Too many failed attempts. Account locked for {settings.LOCKOUT_MINUTES} minutes.",
                headers={"Retry-After": str(lock_seconds)},
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    temp_password_expires_at = _as_utc(employee.temp_password_expires_at)
    if temp_password_expires_at and temp_password_expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Temporary password expired. Contact HR.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    await crud.reset_login_attempts(db, employee.id)
    await crud.update_last_login(db, employee.id)
    employee = await crud.get_by_id(db, employee.id)
    if employee is None or employee.role is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    token = create_access_token(employee, employee.role)
    allowed_domains = _allowed_domains(employee.role.allowed_domains)
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": employee.role.name,
        "allowed_domains": allowed_domains,
        "force_password_change": not employee.password_changed,
        "employee_id": employee.id,
        "full_name": employee.full_name,
    }


@router.post("/logout")
async def logout(current_user: dict = Depends(require_password_changed), token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> dict:
    payload = decode_token(token)
    jti = payload.get("jti")
    exp = payload.get("exp")
    if not isinstance(jti, str):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    if isinstance(exp, int):
        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
    else:
        expires_at = datetime.now(timezone.utc)
    await crud.blacklist_token(db, jti, expires_at)
    return {"message": "Logged out successfully"}


@router.get("/me")
async def me(current_user: dict = Depends(require_password_changed)) -> dict:
    return _serialize_employee(current_user)


@router.post("/change-password")
async def change_password(payload: ChangePasswordRequest, current_user: dict = Depends(get_current_user), token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> dict:
    employee = current_user["employee"]
    if not verify_password(payload.current_password, employee.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Current password is incorrect")

    is_valid, error_message = validate_password_strength(payload.new_password)
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)

    history_hashes = await crud.get_password_history(db, employee.id, limit=3)
    if check_password_history(payload.new_password, history_hashes):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="New password was used previously")

    new_hash = hash_password(payload.new_password)
    await crud.update_password(db, employee.id, new_hash)
    payload_data = decode_token(token)
    jti = payload_data.get("jti")
    exp = payload_data.get("exp")
    if isinstance(jti, str):
        if isinstance(exp, int):
            expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
        else:
            expires_at = datetime.now(timezone.utc)
        await crud.blacklist_token(db, jti, expires_at)

    return {"message": "Password changed. Please login again."}