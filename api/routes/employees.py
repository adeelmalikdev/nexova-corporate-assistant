from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import require_admin, require_hr_admin, require_password_changed
from auth.password import generate_temp_password, hash_password
from config import settings
from database import crud
from database.db import get_db
from database.models import Department, Employee, Role


router = APIRouter(tags=["Employee Management"])


class RoleResponse(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: int
	name: str
	display_name: str
	allowed_domains: list[str]


class DepartmentResponse(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: int
	name: str
	display_name: str


class EmployeeCreate(BaseModel):
	full_name: str = Field(min_length=1)
	email: str
	role_id: int
	department_id: int


class EmployeeUpdate(BaseModel):
	role_id: int | None = None
	department_id: int | None = None


class RoleUpdateRequest(BaseModel):
	role_id: int


class EmployeeResponse(BaseModel):
	model_config = ConfigDict(from_attributes=True)

	id: int
	employee_id: str
	username: str
	email: str
	full_name: str
	role_id: int
	department_id: int
	is_active: bool
	is_verified: bool
	password_changed: bool
	created_by: int | None
	created_at: datetime
	last_login: datetime | None
	failed_login_attempts: int
	locked_until: datetime | None
	temp_password_expires_at: datetime | None


class EmployeeDetailResponse(EmployeeResponse):
	role: RoleResponse | None = None
	department: DepartmentResponse | None = None
	created_by_name: str | None = None


class EmployeeRegisterResponse(BaseModel):
	message: str
	temp_password: str
	employee: EmployeeDetailResponse


class PaginatedEmployeesResponse(BaseModel):
	items: list[EmployeeDetailResponse]
	skip: int
	limit: int
	total: int


def _allowed_domains(raw_allowed_domains: str) -> list[str]:
	try:
		parsed = json.loads(raw_allowed_domains)
	except json.JSONDecodeError:
		return []
	if isinstance(parsed, list):
		return [str(domain) for domain in parsed]
	return []


def _role_response(role: Role | None) -> RoleResponse | None:
	if role is None:
		return None
	return RoleResponse(
		id=role.id,
		name=role.name,
		display_name=role.display_name,
		allowed_domains=_allowed_domains(role.allowed_domains),
	)


def _department_response(department: Department | None) -> DepartmentResponse | None:
	if department is None:
		return None
	return DepartmentResponse(id=department.id, name=department.name, display_name=department.display_name)


def _employee_response(employee: Employee) -> EmployeeDetailResponse:
	return EmployeeDetailResponse(
		id=employee.id,
		employee_id=employee.employee_id,
		username=employee.username,
		email=employee.email,
		full_name=employee.full_name,
		role_id=employee.role_id,
		department_id=employee.department_id,
		is_active=employee.is_active,
		is_verified=employee.is_verified,
		password_changed=employee.password_changed,
		created_by=employee.created_by,
		created_at=employee.created_at,
		last_login=employee.last_login,
		failed_login_attempts=employee.failed_login_attempts,
		locked_until=employee.locked_until,
		temp_password_expires_at=employee.temp_password_expires_at,
		role=_role_response(employee.role),
		department=_department_response(employee.department),
		created_by_name=employee.creator.full_name if employee.creator else None,
	)


def _sanitize_username(full_name: str) -> str:
	parts = re.findall(r"[a-z0-9]+", full_name.lower())
	if not parts:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Full name must contain at least one alphanumeric character",
		)
	return ".".join(parts)


async def _generate_unique_username(db: AsyncSession, full_name: str) -> str:
	base_username = _sanitize_username(full_name)
	if not await crud.username_exists(db, base_username):
		return base_username

	suffix = 2
	while True:
		candidate = f"{base_username}{suffix}"
		if not await crud.username_exists(db, candidate):
			return candidate
		suffix += 1


async def _next_employee_id(db: AsyncSession) -> str:
	current_year = datetime.now(timezone.utc).year
	sequence = await crud.next_employee_sequence(db, current_year)
	return f"NEX-{current_year}-{sequence:03d}"


async def _get_employee_or_404(db: AsyncSession, employee_id: str) -> Employee:
	employee = await crud.get_by_employee_id(db, employee_id)
	if employee is None:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Employee not found")
	return employee


@router.post("/register", response_model=EmployeeRegisterResponse)
async def register_employee(
	payload: EmployeeCreate,
	current_user: dict = Depends(require_hr_admin),
	db: AsyncSession = Depends(get_db),
) -> EmployeeRegisterResponse:
	email = payload.email.strip().lower()
	if not email:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required")
	if await crud.email_exists(db, email):
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")

	role = await crud.get_role_by_id(db, payload.role_id)
	if role is None:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")

	department = await db.get(Department, payload.department_id)
	if department is None:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Department not found")

	username = await _generate_unique_username(db, payload.full_name)
	temp_password = generate_temp_password()
	password_hash = hash_password(temp_password)
	employee_id = await _next_employee_id(db)
	created_by_employee = current_user.get("employee")
	created_by_id = created_by_employee.id if created_by_employee is not None else None
	expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.TEMP_PASSWORD_EXPIRY_HOURS)

	employee = Employee(
		employee_id=employee_id,
		username=username,
		email=email,
		full_name=payload.full_name,
		password_hash=password_hash,
		role_id=role.id,
		department_id=department.id,
		is_active=True,
		is_verified=False,
		password_changed=False,
		created_by=created_by_id,
		temp_password_expires_at=expires_at,
	)
	db.add(employee)
	await db.commit()
	await db.refresh(employee)
	stored_employee = await _get_employee_or_404(db, employee.employee_id)
	return EmployeeRegisterResponse(
		message="Share this temporary password with the employee. It expires in 48 hours.",
		temp_password=temp_password,
		employee=_employee_response(stored_employee),
	)


@router.get("", response_model=PaginatedEmployeesResponse)
async def list_employees(
	skip: int = Query(0, ge=0),
	limit: int = Query(100, ge=1, le=500),
	current_user: dict = Depends(require_admin),
	db: AsyncSession = Depends(get_db),
) -> PaginatedEmployeesResponse:
	total_result = await db.execute(select(func.count(Employee.id)))
	total = total_result.scalar_one()
	items = await crud.get_all(db, skip=skip, limit=limit)
	return PaginatedEmployeesResponse(
		items=[_employee_response(employee) for employee in items],
		skip=skip,
		limit=limit,
		total=total,
	)


@router.get("/pending", response_model=list[EmployeeDetailResponse])
async def pending_employees(
	current_user: dict = Depends(require_hr_admin),
	db: AsyncSession = Depends(get_db),
) -> list[EmployeeDetailResponse]:
	items = await crud.get_pending(db)
	return [_employee_response(employee) for employee in items]


@router.get("/roles", response_model=list[RoleResponse])
async def list_roles(
	current_user: dict = Depends(require_password_changed),
	db: AsyncSession = Depends(get_db),
) -> list[RoleResponse]:
	roles = await crud.get_all_roles(db)
	return [
		RoleResponse(
			id=role.id,
			name=role.name,
				display_name=role.display_name,
				allowed_domains=_allowed_domains(role.allowed_domains),
		)
		for role in roles
	]


@router.get("/departments", response_model=list[DepartmentResponse])
async def list_departments(
	current_user: dict = Depends(require_password_changed),
	db: AsyncSession = Depends(get_db),
) -> list[DepartmentResponse]:
	departments = await crud.get_all_departments(db)
	return [
		DepartmentResponse(id=department.id, name=department.name, display_name=department.display_name)
		for department in departments
	]


@router.get("/{employee_id}", response_model=EmployeeDetailResponse)
async def get_employee(
	employee_id: str,
	current_user: dict = Depends(require_password_changed),
	db: AsyncSession = Depends(get_db),
) -> EmployeeDetailResponse:
	employee = await _get_employee_or_404(db, employee_id)
	current_employee = current_user["employee"]
	if current_user.get("role_name") != "admin" and employee.id != current_employee.id:
		raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
	return _employee_response(employee)


@router.patch("/{employee_id}/verify", response_model=EmployeeDetailResponse)
async def verify_employee(
	employee_id: str,
	current_user: dict = Depends(require_hr_admin),
	db: AsyncSession = Depends(get_db),
) -> EmployeeDetailResponse:
	employee = await _get_employee_or_404(db, employee_id)
	if employee.is_verified:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Employee already verified")
	employee.is_verified = True
	await db.commit()
	await db.refresh(employee)
	return _employee_response(await _get_employee_or_404(db, employee_id))


@router.patch("/{employee_id}/deactivate", response_model=EmployeeDetailResponse)
async def deactivate_employee(
	employee_id: str,
	current_user: dict = Depends(require_admin),
	db: AsyncSession = Depends(get_db),
) -> EmployeeDetailResponse:
	employee = await _get_employee_or_404(db, employee_id)
	if employee.id == current_user["employee"].id:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot deactivate your own account")
	employee.is_active = False
	await db.commit()
	await db.refresh(employee)
	return _employee_response(await _get_employee_or_404(db, employee_id))


@router.patch("/{employee_id}/role", response_model=EmployeeDetailResponse)
async def change_employee_role(
	employee_id: str,
	payload: RoleUpdateRequest,
	current_user: dict = Depends(require_admin),
	db: AsyncSession = Depends(get_db),
) -> EmployeeDetailResponse:
	employee = await _get_employee_or_404(db, employee_id)
	if employee.id == current_user["employee"].id:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot change your own role")
	role = await crud.get_role_by_id(db, payload.role_id)
	if role is None:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Role not found")
	employee.role_id = role.id
	await db.commit()
	await db.refresh(employee)
	return _employee_response(await _get_employee_or_404(db, employee_id))