from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import HTTPException, status
from jose import jwt
from jose.exceptions import ExpiredSignatureError, JWTError

from config import Settings
from database.models import Employee, Role


settings = Settings()


def _parse_allowed_domains(raw_allowed_domains: str) -> list[str]:
	try:
		parsed = json.loads(raw_allowed_domains)
	except json.JSONDecodeError:
		return []

	if isinstance(parsed, list):
		return [str(domain) for domain in parsed]
	return []


def create_access_token(employee: Employee, role: Role) -> str:
	now = datetime.now(timezone.utc)
	payload = {
		"sub": employee.username,
		"employee_id": employee.id,
		"role": role.name,
		"allowed_domains": _parse_allowed_domains(role.allowed_domains),
		"department_id": employee.department_id,
		"force_password_change": not employee.password_changed,
		"jti": str(uuid4()),
		"exp": now + timedelta(minutes=settings.JWT_EXPIRY_MINUTES),
		"iat": now,
	}
	return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
	try:
		return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
	except (ExpiredSignatureError, JWTError) as exc:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid or expired token",
			headers={"WWW-Authenticate": "Bearer"},
		) from exc


def extract_jti(token: str) -> str:
	payload = decode_token(token)
	jti = payload.get("jti")
	if not isinstance(jti, str) or not jti:
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Invalid or expired token",
			headers={"WWW-Authenticate": "Bearer"},
		)
	return jti
