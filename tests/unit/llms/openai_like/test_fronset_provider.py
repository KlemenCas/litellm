import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_fronset",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "3.press-release-claims",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Revenue: $4.2 billion, up 12%"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
}

_PRESS_RELEASE: Final = "Acme Corp reported third-quarter revenue of $4.2 billion, up 12% year over year"

_SUMMARY_VARIABLES: Final = {
    "content_text": _PRESS_RELEASE,
    "content_title": "Acme Corp Q3 results",
    "content_type": "press release",
    "content_url": "",
    "summary_structure": "## Figures\nRevenue and growth",
}


def test_fronset_is_a_registered_provider():
    assert litellm.LlmProviders.FRONSET.value == "fronset"
    assert "fronset" in litellm.provider_list
    assert "fronset" in litellm.constants.openai_compatible_providers


def test_fronset_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("FRONSET_API_KEY", "llmb_i_test")

    model, provider, api_key, api_base = get_llm_provider(
        model="fronset/3.press-release-claims",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "3.press-release-claims"
    assert provider == "fronset"
    assert api_key == "llmb_i_test"
    assert api_base == "https://api.fronset.ai/v1"


def test_fronset_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("FRONSET_API_KEY", "llmb_i_env")
    monkeypatch.setenv("FRONSET_API_BASE", "https://fronset.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="fronset/3.press-release-claims",
        custom_llm_provider=None,
        api_base="https://fronset.internal.example/v1",
        api_key="llmb_i_explicit",
    )

    assert provider == "fronset"
    assert api_key == "llmb_i_explicit"
    assert api_base == "https://fronset.internal.example/v1"


def test_fronset_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("FRONSET_API_KEY", "llmb_i_env")
    monkeypatch.setenv("FRONSET_API_BASE", "https://fronset.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="fronset/3.press-release-claims",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "fronset"
    assert api_base == "https://fronset.env.example/v1"


def test_fronset_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("FRONSET_API_KEY", "llmb_i_env")

    model, provider, api_key, api_base = get_llm_provider(
        model="3.press-release-claims",
        custom_llm_provider=None,
        api_base="https://api.fronset.ai/v1",
        api_key=None,
    )

    assert model == "3.press-release-claims"
    assert provider == "fronset"
    assert api_key == "llmb_i_env"
    assert api_base == "https://api.fronset.ai/v1"


def test_fronset_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    fronset = next(provider for provider in providers if provider["litellm_provider"] == "fronset")

    assert fronset["provider"] == "FRONSET"
    assert fronset["provider_display_name"] == "Fronset"
    assert fronset["default_model_placeholder"] == "fronset/structured-content-summarization"
    assert {field["key"]: field["required"] for field in fronset["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_fronset_supported_endpoints():
    backup_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    for path in (backup_path, root_path):
        endpoints = json.loads(path.read_text())["providers"]["fronset"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert not any(value for key, value in endpoints.items() if key != "chat_completions")


def test_fronset_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.fronset.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="fronset/3.press-release-claims",
            messages=[{"role": "user", "content": _PRESS_RELEASE}],
            api_key="llmb_i_test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.fronset.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer llmb_i_test"
    assert body["model"] == "3.press-release-claims"
    assert body["messages"] == [{"role": "user", "content": _PRESS_RELEASE}]
    assert response.choices[0].message.content == "Revenue: $4.2 billion, up 12%"


def test_fronset_template_task_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.fronset.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="fronset/structured-content-summarization",
            messages=[],
            extra_body={"llmbench": {"prompt_vars": _SUMMARY_VARIABLES}},
            api_key="llmb_i_test",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["model"] == "structured-content-summarization"
    assert body["messages"] == []
    assert body["llmbench"] == {"prompt_vars": _SUMMARY_VARIABLES}
