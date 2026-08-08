"""Canonical, dependency-light schema for one visual compliance sample."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


class SchemaError(ValueError):
    """Raised when serialized data violates the canonical sample contract."""


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class BoundingBox:
    """An integer pixel box in ``(x1, y1, x2, y2)`` coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise SchemaError("bbox coordinates must be integers")
        if self.x1 < 0 or self.y1 < 0:
            raise SchemaError("bbox coordinates must be non-negative")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise SchemaError("bbox must satisfy x1 < x2 and y1 < y2")

    @classmethod
    def from_value(cls, value: Sequence[Any]) -> "BoundingBox":
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
            raise SchemaError("bbox must contain exactly four coordinates")
        return cls(*value)

    def within(self, width: int, height: int) -> bool:
        return self.x2 <= width and self.y2 <= height

    def as_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class ObjectAnnotation:
    label: str
    bbox: BoundingBox

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _required_text(self.label, "object.label"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectAnnotation":
        if not isinstance(value, Mapping):
            raise SchemaError("response.objects entries must be objects")
        return cls(
            label=_required_text(value.get("label"), "object.label"),
            bbox=BoundingBox.from_value(value.get("bbox")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "bbox": self.bbox.as_list()}


@dataclass(frozen=True)
class SampleResponse:
    result: str
    objects: tuple[ObjectAnnotation, ...]
    reason: str
    uncertain: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "result", _required_text(self.result, "response.result"))
        object.__setattr__(self, "reason", _required_text(self.reason, "response.reason"))
        if not isinstance(self.uncertain, bool):
            raise SchemaError("response.uncertain must be a boolean")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SampleResponse":
        if not isinstance(value, Mapping):
            raise SchemaError("response must be an object")
        raw_objects = value.get("objects")
        if not isinstance(raw_objects, list):
            raise SchemaError("response.objects must be an array")
        return cls(
            result=_required_text(value.get("result"), "response.result"),
            objects=tuple(ObjectAnnotation.from_dict(item) for item in raw_objects),
            reason=_required_text(value.get("reason"), "response.reason"),
            uncertain=value.get("uncertain", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result,
            "objects": [item.to_dict() for item in self.objects],
            "reason": self.reason,
            "uncertain": self.uncertain,
        }


@dataclass(frozen=True)
class LicenseInfo:
    """Per-sample license provenance required before a dataset release."""

    identifier: str
    name: str
    url: str | None = None
    attribution: str | None = None
    redistributable: bool | None = None
    commercial_use: bool | None = None
    derivative_work: bool | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifier", _required_text(self.identifier, "license.identifier"))
        object.__setattr__(self, "name", _required_text(self.name, "license.name"))
        if self.url is not None and not self.url.startswith(("https://", "http://")):
            raise SchemaError("license.url must use http or https")
        for field_name in ("attribution", "notes"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise SchemaError(f"license.{field_name} must be a string or null")
        for field_name in ("redistributable", "commercial_use", "derivative_work"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, bool):
                raise SchemaError(f"license.{field_name} must be a boolean or null")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LicenseInfo":
        if not isinstance(value, Mapping):
            raise SchemaError("license must be an object with explicit provenance fields")
        return cls(
            identifier=value.get("identifier"),
            name=value.get("name"),
            url=value.get("url"),
            attribution=value.get("attribution"),
            redistributable=value.get("redistributable"),
            commercial_use=value.get("commercial_use"),
            derivative_work=value.get("derivative_work"),
            notes=value.get("notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "identifier": self.identifier,
                "name": self.name,
                "url": self.url,
                "attribution": self.attribution,
                "redistributable": self.redistributable,
                "commercial_use": self.commercial_use,
                "derivative_work": self.derivative_work,
                "notes": self.notes,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class DataSample:
    sample_id: str
    entity_id: str
    image: str
    instruction: str
    response: SampleResponse
    difficulty: tuple[str, ...]
    source: str
    license: LicenseInfo
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _required_text(self.sample_id, "sample_id"))
        object.__setattr__(self, "entity_id", _required_text(self.entity_id, "entity_id"))
        object.__setattr__(self, "instruction", _required_text(self.instruction, "instruction"))
        object.__setattr__(self, "source", _required_text(self.source, "source"))
        image = _required_text(self.image, "image")
        path = PurePosixPath(image)
        if path.is_absolute() or ".." in path.parts:
            raise SchemaError("image must be a relative path without parent traversal")
        object.__setattr__(self, "image", image)
        if not isinstance(self.difficulty, tuple) or any(
            not isinstance(tag, str) or not tag.strip() for tag in self.difficulty
        ):
            raise SchemaError("difficulty must contain non-empty strings")
        if len(set(self.difficulty)) != len(self.difficulty):
            raise SchemaError("difficulty tags must be unique")
        if not isinstance(self.metadata, Mapping):
            raise SchemaError("metadata must be an object")

    @property
    def is_hard_negative(self) -> bool:
        return "hard_negative" in self.difficulty

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataSample":
        if not isinstance(value, Mapping):
            raise SchemaError("sample must be an object")
        raw_difficulty = value.get("difficulty", [])
        if not isinstance(raw_difficulty, list):
            raise SchemaError("difficulty must be an array")
        return cls(
            sample_id=value.get("sample_id"),
            entity_id=value.get("entity_id"),
            image=value.get("image"),
            instruction=value.get("instruction"),
            response=SampleResponse.from_dict(value.get("response")),
            difficulty=tuple(raw_difficulty),
            source=value.get("source"),
            license=LicenseInfo.from_dict(value.get("license")),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "entity_id": self.entity_id,
            "image": self.image,
            "instruction": self.instruction,
            "response": self.response.to_dict(),
            "difficulty": list(self.difficulty),
            "source": self.source,
            "license": self.license.to_dict(),
            "metadata": dict(self.metadata),
        }
