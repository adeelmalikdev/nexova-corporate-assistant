from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from config import Settings
from database.models import Department, Employee, InvalidatedToken, PasswordHistory, Role


settings = Settings()


def _employee_options() -> list:
	return [
		selectinload(Employee.role),
		selectinload(Employee.department),
		selectinload(Employee.creator),
	]


async def get_by_employee_id(db: AsyncSession, employee_id: str) -> Employee | None:
	result = await db.execute(
		select(Employee)
		.options(*_employee_options())
		.where(Employee.employee_id == employee_id)
	)
	return result.scalar_one_or_none()


async def get_by_id(db: AsyncSession, id: int) -> Employee | None:
	result = await db.execute(
		select(Employee)
		.options(*_employee_options())
		.where(Employee.id == id)
	)
	return result.scalar_one_or_none()


async def get_by_username(db: AsyncSession, username: str) -> Employee | None:
	result = await db.execute(
		select(Employee)
		.options(*_employee_options())
		.where(Employee.username == username)
	)
	return result.scalar_one_or_none()


async def get_by_email(db: AsyncSession, email: str) -> Employee | None:
	result = await db.execute(
		select(Employee)
		.options(*_employee_options())
		.where(Employee.email == email)
	)
	return result.scalar_one_or_none()


async def username_exists(db: AsyncSession, username: str) -> bool:
	result = await db.execute(select(Employee.id).where(func.lower(Employee.username) == username.lower()))
	return result.scalar_one_or_none() is not None


async def email_exists(db: AsyncSession, email: str) -> bool:
	result = await db.execute(select(Employee.id).where(func.lower(Employee.email) == email.lower()))
	return result.scalar_one_or_none() is not None


async def get_role_by_id(db: AsyncSession, role_id: int) -> Role | None:
	result = await db.execute(select(Role).where(Role.id == role_id))
	return result.scalar_one_or_none()


async def get_all_roles(db: AsyncSession) -> list[Role]:
	result = await db.execute(select(Role).order_by(Role.id))
	return list(result.scalars().all())


async def get_all_departments(db: AsyncSession) -> list[Department]:
	result = await db.execute(select(Department).order_by(Department.id))
	return list(result.scalars().all())


async def next_employee_sequence(db: AsyncSession, year: int) -> int:
	result = await db.execute(select(Employee.employee_id).where(Employee.employee_id.like(f"NEX-{year}-%")))
	max_sequence = 0
	pattern = re.compile(rf"^NEX-{year}-(\d+)$")
	for employee_code in result.scalars().all():
		match = pattern.match(employee_code)
		if match:
			max_sequence = max(max_sequence, int(match.group(1)))
	return max_sequence + 1


async def get_all(db: AsyncSession, skip: int = 0, limit: int = 100) -> list[Employee]:
	result = await db.execute(
		select(Employee)
		.options(*_employee_options())
		.order_by(Employee.id)
		.offset(skip)
		.limit(limit)
	)
	return list(result.scalars().all())


async def get_pending(db: AsyncSession) -> list[Employee]:
	result = await db.execute(
		select(Employee)
		.options(*_employee_options())
		.where(Employee.is_verified.is_(False))
		.order_by(Employee.id)
	)
	return list(result.scalars().all())


async def create(db: AsyncSession, data: dict) -> Employee:
	current_year = datetime.now(timezone.utc).year
	employee_seq = data.get("employee_seq")
	if employee_seq is None:
		employee_seq = await next_employee_sequence(db, current_year)
	employee_code = f"NEX-{current_year}-{employee_seq:03d}"

	data = {key: value for key, value in data.items() if key != "employee_seq"}
	employee = Employee(employee_id=employee_code, **data)
	db.add(employee)
	await db.commit()
	await db.refresh(employee)
	result = await db.execute(
		select(Employee)
		.options(*_employee_options())
		.where(Employee.id == employee.id)
	)
	employee = result.scalar_one()
	return employee


async def verify(db: AsyncSession, employee_id: int) -> Employee:
	employee = await get_by_id(db, employee_id)
	if employee is None:
		raise ValueError("Employee not found")
	employee.is_verified = True
	await db.commit()
	await db.refresh(employee)
	return employee


async def deactivate(db: AsyncSession, employee_id: int) -> Employee:
	employee = await get_by_id(db, employee_id)
	if employee is None:
		raise ValueError("Employee not found")
	employee.is_active = False
	await db.commit()
	await db.refresh(employee)
	return employee


async def update_role(db: AsyncSession, employee_id: int, role_id: int) -> Employee:
	employee = await get_by_id(db, employee_id)
	if employee is None:
		raise ValueError("Employee not found")
	employee.role_id = role_id
	await db.commit()
	await db.refresh(employee)
	return employee


async def record_failed_login(db: AsyncSession, employee_id: int) -> None:
	employee = await get_by_id(db, employee_id)
	if employee is None:
		raise ValueError("Employee not found")

	employee.failed_login_attempts += 1
	if employee.failed_login_attempts >= settings.MAX_LOGIN_ATTEMPTS:
		employee.locked_until = datetime.now(timezone.utc) + timedelta(minutes=settings.LOCKOUT_MINUTES)

	await db.commit()


async def reset_login_attempts(db: AsyncSession, employee_id: int) -> None:
	employee = await get_by_id(db, employee_id)
	if employee is None:
		raise ValueError("Employee not found")

	employee.failed_login_attempts = 0
	employee.locked_until = None
	await db.commit()


async def update_last_login(db: AsyncSession, employee_id: int) -> None:
	employee = await get_by_id(db, employee_id)
	if employee is None:
		raise ValueError("Employee not found")

	employee.last_login = datetime.now(timezone.utc)
	await db.commit()


async def update_password(db: AsyncSession, employee_id: int, new_hash: str) -> Employee:
	employee = await get_by_id(db, employee_id)
	if employee is None:
		raise ValueError("Employee not found")

	old_hash = employee.password_hash
	if old_hash:
		db.add(PasswordHistory(employee_id=employee.id, password_hash=old_hash))

	employee.password_hash = new_hash
	employee.password_changed = True
	employee.temp_password_expires_at = None
	await db.commit()
	await db.refresh(employee)
	return employee


async def get_password_history(db: AsyncSession, employee_id: int, limit: int = 3) -> list[str]:
	result = await db.execute(
		select(PasswordHistory.password_hash)
		.where(PasswordHistory.employee_id == employee_id)
		.order_by(PasswordHistory.created_at.desc())
		.limit(limit)
	)
	return [row[0] for row in result.all()]


async def blacklist_token(db: AsyncSession, jti: str, expires_at: datetime) -> None:
	if await is_token_blacklisted(db, jti):
		return
	db.add(InvalidatedToken(jti=jti, expires_at=expires_at))
	await db.commit()


async def is_token_blacklisted(db: AsyncSession, jti: str) -> bool:
	result = await db.execute(select(InvalidatedToken.id).where(InvalidatedToken.jti == jti))
	return result.scalar_one_or_none() is not None


async def cleanup_expired_tokens(db: AsyncSession) -> None:
	await db.execute(delete(InvalidatedToken).where(InvalidatedToken.expires_at < datetime.now(timezone.utc)))
	await db.commit()
