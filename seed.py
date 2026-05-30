from __future__ import annotations
import asyncio
import json
from database.db import async_session, init_db
from database.models import Department, Role, Employee
from auth.password import hash_password
from sqlalchemy import select


async def seed() -> None:
    async with async_session() as db:

        # Departments
        departments = [
            {"name": "hr", "display_name": "Human Resources"},
            {"name": "legal", "display_name": "Legal & Compliance"},
            {"name": "finance", "display_name": "Finance"},
            {"name": "engineering", "display_name": "Engineering"},
        ]
        for d in departments:
            exists = await db.execute(select(Department).where(Department.name == d["name"]))
            if not exists.scalar_one_or_none():
                db.add(Department(**d))
        await db.commit()

        # Roles
        roles = [
            {"name": "employee", "display_name": "Employee", "allowed_domains": json.dumps(["hr"])},
            {"name": "manager", "display_name": "Manager", "allowed_domains": json.dumps(["hr"])},
            {"name": "dept_head", "display_name": "Department Head", "allowed_domains": json.dumps(["hr", "legal"])},
            {"name": "hr_admin", "display_name": "HR Admin", "allowed_domains": json.dumps(["hr", "finance"])},
            {"name": "admin", "display_name": "Administrator", "allowed_domains": json.dumps(["hr", "legal", "finance", "engineering"])},
        ]
        for r in roles:
            exists = await db.execute(select(Role).where(Role.name == r["name"]))
            if not exists.scalar_one_or_none():
                db.add(Role(**r))
        await db.commit()

        # Admin user
        exists = await db.execute(select(Employee).where(Employee.username == "admin"))
        if not exists.scalar_one_or_none():
            admin_role = await db.execute(select(Role).where(Role.name == "admin"))
            role = admin_role.scalar_one()
            hr_dept = await db.execute(select(Department).where(Department.name == "hr"))
            dept = hr_dept.scalar_one()
            db.add(Employee(
                employee_id="NEX-2025-000",
                username="admin",
                email="admin@nexova.io",
                full_name="System Administrator",
                password_hash=hash_password("Admin@nexova1"),
                role_id=role.id,
                department_id=dept.id,
                is_active=True,
                is_verified=True,
                password_changed=True,
                failed_login_attempts=0,
            ))
            await db.commit()
            print("✓ Admin user created")
        else:
            print("✓ Admin user already exists")


def run_seed() -> None:
    asyncio.run(seed())


if __name__ == "__main__":
    run_seed()