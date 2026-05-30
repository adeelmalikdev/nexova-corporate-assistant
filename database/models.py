from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database.db import Base


class Department(Base):
	__tablename__ = "departments"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
	name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
	display_name: Mapped[str] = mapped_column(String(120), nullable=False)

	employees: Mapped[list[Employee]] = relationship(back_populates="department")


class Role(Base):
	__tablename__ = "roles"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
	name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
	display_name: Mapped[str] = mapped_column(String(120), nullable=False)
	allowed_domains: Mapped[str] = mapped_column(Text, nullable=False)

	employees: Mapped[list[Employee]] = relationship(back_populates="role")


class Employee(Base):
	__tablename__ = "employees"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
	employee_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
	username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
	email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
	full_name: Mapped[str] = mapped_column(String(255), nullable=False)
	password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
	role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), nullable=False)
	department_id: Mapped[int] = mapped_column(ForeignKey("departments.id"), nullable=False)
	is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
	is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
	password_changed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
	created_by: Mapped[int | None] = mapped_column(ForeignKey("employees.id"), nullable=True)
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
	last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
	failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
	locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
	temp_password_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

	role: Mapped[Role] = relationship(back_populates="employees")
	department: Mapped[Department] = relationship(back_populates="employees")
	creator: Mapped[Employee | None] = relationship(remote_side="Employee.id", foreign_keys=[created_by])
	password_history: Mapped[list[PasswordHistory]] = relationship(back_populates="employee", cascade="all, delete-orphan")


class InvalidatedToken(Base):
	__tablename__ = "invalidated_tokens"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
	jti: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
	expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
	invalidated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PasswordHistory(Base):
	__tablename__ = "password_history"

	id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
	employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), nullable=False, index=True)
	password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

	employee: Mapped[Employee] = relationship(back_populates="password_history")

