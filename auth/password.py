from __future__ import annotations

import secrets
import string

from passlib.context import CryptContext


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SPECIAL_CHARACTERS = "!@#$%^&*"


def hash_password(plain: str) -> str:
	return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
	return pwd_context.verify(plain, hashed)


def validate_password_strength(plain: str) -> tuple[bool, str]:
	if len(plain) < 8:
		return False, "Password must be at least 8 characters long"
	if not any(character.isupper() for character in plain):
		return False, "Password must contain at least 1 uppercase letter"
	if not any(character.islower() for character in plain):
		return False, "Password must contain at least 1 lowercase letter"
	if not any(character.isdigit() for character in plain):
		return False, "Password must contain at least 1 digit"
	if not any(character in SPECIAL_CHARACTERS for character in plain):
		return False, "Password must contain at least 1 special character (!@#$%^&*)"
	return True, "Password is strong enough"


def check_password_history(new_plain: str, history_hashes: list[str]) -> bool:
	for history_hash in history_hashes[:3]:
		try:
			if verify_password(new_plain, history_hash):
				return True
		except Exception:
			continue
	return False


def generate_temp_password() -> str:
	digits = "".join(secrets.choice(string.digits) for _ in range(4))
	uppercase = "".join(secrets.choice(string.ascii_uppercase) for _ in range(3))
	lowercase = "".join(secrets.choice(string.ascii_lowercase) for _ in range(3))
	return f"Nex@{digits}{uppercase}{lowercase}"
