"""Google OpenAPI schema conversion for function declarations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ._typing import is_string_mapping


def _schema_sequence(value: object) -> list[object] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    return list(value)


def _empty_object_schema(schema: Mapping[str, object]) -> bool:
    properties = schema.get("properties")
    no_properties = not isinstance(properties, Mapping) or not properties
    additional_properties = schema.get("additionalProperties")
    return (
        schema.get("type") == "object" and no_properties and additional_properties in (None, False)
    )


def google_openapi_schema(schema: object, *, _root: bool = True) -> dict[str, object] | None:
    """Convert JSON Schema Draft 7 to Google's OpenAPI 3.0 subset."""

    if schema is None:
        return None
    if isinstance(schema, bool):
        return {"type": "boolean", "properties": {}}
    if not is_string_mapping(schema):
        return None
    if _empty_object_schema(schema):
        if _root:
            return None
        nested: dict[str, object] = {"type": "object"}
        description = schema.get("description")
        if description:
            nested["description"] = description
        return nested

    result: dict[str, object] = {}
    for key in ("description", "required", "format"):
        value = schema.get(key)
        if value:
            result[key] = value

    if "const" in schema:
        result["enum"] = [schema["const"]]

    raw_type = schema.get("type")
    types = _schema_sequence(raw_type)
    if types is not None:
        has_null = "null" in types
        non_null_types = [value for value in types if value != "null"]
        if not non_null_types:
            result["type"] = "null"
        else:
            result["anyOf"] = [{"type": value} for value in non_null_types]
            if has_null:
                result["nullable"] = True
    elif raw_type:
        result["type"] = raw_type

    if "enum" in schema:
        result["enum"] = schema["enum"]

    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        converted_properties: dict[str, object] = {}
        for key, value in properties.items():
            if isinstance(key, str) and isinstance(value, (Mapping, bool)):
                converted = google_openapi_schema(value, _root=False)
                if converted is not None:
                    converted_properties[key] = converted
        result["properties"] = converted_properties

    raw_items = schema.get("items")
    items = _schema_sequence(raw_items)
    if items is not None:
        converted_items: list[object] = []
        for item in items:
            if isinstance(item, (Mapping, bool)):
                converted = google_openapi_schema(item, _root=False)
                if converted is not None:
                    converted_items.append(converted)
        result["items"] = converted_items
    elif isinstance(raw_items, (Mapping, bool)):
        converted = google_openapi_schema(raw_items, _root=False)
        if converted is not None:
            result["items"] = converted

    for keyword in ("allOf", "oneOf"):
        branches = _schema_sequence(schema.get(keyword))
        if branches is None:
            continue
        converted_branches: list[object] = []
        for branch in branches:
            if isinstance(branch, (Mapping, bool)):
                converted = google_openapi_schema(branch, _root=False)
                if converted is not None:
                    converted_branches.append(converted)
        result[keyword] = converted_branches

    any_of = _schema_sequence(schema.get("anyOf"))
    if any_of is not None:
        non_null = [
            branch
            for branch in any_of
            if not (isinstance(branch, Mapping) and branch.get("type") == "null")
        ]
        nullable = len(non_null) != len(any_of)
        converted_any_of = [
            converted
            for branch in non_null
            if isinstance(branch, (Mapping, bool))
            and (converted := google_openapi_schema(branch, _root=False)) is not None
        ]
        if nullable and len(converted_any_of) == 1:
            result["nullable"] = True
            result.update(converted_any_of[0])
        else:
            result["anyOf"] = converted_any_of
            if nullable:
                result["nullable"] = True

    if "minLength" in schema:
        result["minLength"] = schema["minLength"]

    return result


__all__ = ["google_openapi_schema"]
