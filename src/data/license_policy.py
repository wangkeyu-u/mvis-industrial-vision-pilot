"""Explicit license allow/deny policy for dataset imports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import LicenseInfo, SchemaError


@dataclass(frozen=True)
class LicenseDecision:
    allowed: bool
    code: str
    message: str


@dataclass(frozen=True)
class LicensePolicy:
    allowed_identifiers: frozenset[str]
    denied_identifiers: frozenset[str]
    require_explicit_identifier: bool = True

    def __post_init__(self) -> None:
        overlap = self.allowed_identifiers & self.denied_identifiers
        if overlap:
            raise ValueError(f"license identifiers cannot be both allowed and denied: {sorted(overlap)}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LicensePolicy":
        allow = value.get("allow", {})
        deny = value.get("deny", {})
        if not isinstance(allow, Mapping) or not isinstance(deny, Mapping):
            raise ValueError("license policy allow and deny sections must be objects")
        allowed = allow.get("identifiers", [])
        denied = deny.get("identifiers", [])
        if not isinstance(allowed, list) or not isinstance(denied, list):
            raise ValueError("license identifier lists must be arrays")
        if any(not isinstance(item, str) or not item.strip() for item in allowed + denied):
            raise ValueError("license identifiers must be non-empty strings")
        require_explicit = value.get("require_explicit_identifier", True)
        if not isinstance(require_explicit, bool):
            raise ValueError("require_explicit_identifier must be a boolean")
        return cls(frozenset(allowed), frozenset(denied), require_explicit)

    @classmethod
    def load(cls, path: str | Path) -> "LicensePolicy":
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("license policy must be a YAML object")
        return cls.from_dict(value)

    def decide(self, license_info: LicenseInfo) -> LicenseDecision:
        identifier = license_info.identifier.strip().lower()
        denied = {item.lower() for item in self.denied_identifiers}
        allowed = {item.lower() for item in self.allowed_identifiers}
        if identifier in denied:
            return LicenseDecision(False, "LICENSE_DENIED", f"license {identifier!r} is denied")
        if allowed and identifier not in allowed:
            return LicenseDecision(
                False,
                "LICENSE_NOT_ALLOWLISTED",
                f"license {identifier!r} is not in the allowlist",
            )
        if self.require_explicit_identifier and not identifier:
            return LicenseDecision(False, "LICENSE_MISSING", "license identifier is required")
        return LicenseDecision(True, "LICENSE_ALLOWED", f"license {identifier!r} is allowed")


def require_allowed_license(policy: LicensePolicy, license_info: LicenseInfo) -> None:
    decision = policy.decide(license_info)
    if not decision.allowed:
        raise SchemaError(f"{decision.code}: {decision.message}")
