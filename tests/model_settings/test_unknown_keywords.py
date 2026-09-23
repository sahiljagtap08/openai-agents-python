"""Typed settings constructors reject misspelled keyword arguments.

The dictionary forms already raise on unknown keys. The typed constructors are pydantic
dataclasses, which drop unknown keywords by default, so a typo such as
``ModelSettings(max_output_tokens=100)`` used to produce a settings object that silently
carried none of the intended configuration.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agents.memory import SessionSettings
from agents.model_settings import MCPToolChoice, ModelSettings
from agents.retry import ModelRetryBackoffSettings, ModelRetrySettings


@pytest.mark.parametrize(
    ("factory", "keyword"),
    [
        (ModelSettings, "temperatur"),
        (ModelSettings, "max_output_tokens"),
        (MCPToolChoice, "server_lable"),
        (ModelRetrySettings, "max_retry"),
        (ModelRetryBackoffSettings, "initial_delai"),
        (SessionSettings, "limitt"),
    ],
)
def test_typed_settings_reject_unknown_keywords(factory: type, keyword: str) -> None:
    with pytest.raises(ValidationError, match=keyword):
        factory(**{keyword: 1})


def test_typed_settings_still_accept_known_fields() -> None:
    settings = ModelSettings(
        max_tokens=100,
        retry=ModelRetrySettings(max_retries=2, backoff=ModelRetryBackoffSettings(initial_delay=1)),
    )
    assert settings.max_tokens == 100
    assert settings.retry is not None
    assert settings.retry.max_retries == 2
    assert SessionSettings(limit=3).limit == 3
    assert MCPToolChoice(server_label="srv", name="tool").name == "tool"
