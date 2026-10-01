"""Bound provider inputs without shortening identifiers or user intent."""
from app.config import config


class PromptBudgetError(Exception):
    """The complete prompt cannot fit the configured input budget."""


def shorten_value(value):
    if isinstance(value, str) and len(value) > config.max_sample_value_chars:
        return value[:config.max_sample_value_chars] + "… [sample shortened]"
    return value


def bounded_profile_json(profile):
    data = profile.model_dump(mode="json")
    import json
    shortened = False
    for row in data["sample_rows"]:
        for key, value in row.items():
            bounded = shorten_value(value)
            shortened |= bounded != value
            row[key] = bounded
    # Large text stats can be supplied by custom profiles; preserve all schema identifiers.
    for column in data["columns"]:
        for key in ("min", "max"):
            bounded = shorten_value(column[key])
            shortened |= bounded != column[key]
            column[key] = bounded
    if shortened:
        data["sample_notice"] = "Long sample values were shortened; SQL still queries the full data."
    return json.dumps(data, indent=2, ensure_ascii=False)


def ensure_prompt_size(prompt, system_prompt=None):
    if len(prompt) + len(system_prompt or "") > config.max_prompt_chars:
        raise PromptBudgetError(
            "This question and schema exceed the supported model input size. "
            "Use a shorter question or a dataset with fewer columns."
        )
