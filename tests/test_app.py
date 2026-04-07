from __future__ import annotations

import pytest

from rleditor.app import _resolve_initial_plugin_id
from rleditor.core.models import TaskDefinition
from rleditor.plugins.base import EnvironmentPlugin
from rleditor.plugins.registry import PluginRegistry


class _DummyBackend:
    def default_task(self) -> TaskDefinition:
        return TaskDefinition(environment_id="dummy_env", name="Dummy Task")

    def create_env(self, task: TaskDefinition):
        _ = task
        return object()


def _register_dummy_plugin(registry: PluginRegistry, *, plugin_id: str) -> None:
    registry.register_environment(
        EnvironmentPlugin(
            plugin_id=plugin_id,
            display_name=plugin_id.title(),
            description="Test plugin",
            backend=_DummyBackend(),
            gui_extension=None,
        )
    )


def test_resolve_initial_plugin_id_returns_requested_plugin() -> None:
    registry = PluginRegistry()
    _register_dummy_plugin(registry, plugin_id="dummy")

    assert _resolve_initial_plugin_id(registry, "dummy") == "dummy"


def test_resolve_initial_plugin_id_defaults_when_only_one_plugin_exists() -> None:
    registry = PluginRegistry()
    _register_dummy_plugin(registry, plugin_id="dummy")

    assert _resolve_initial_plugin_id(registry, None) == "dummy"


def test_resolve_initial_plugin_id_requires_explicit_choice_when_multiple_plugins_exist() -> None:
    registry = PluginRegistry()
    _register_dummy_plugin(registry, plugin_id="dummy_a")
    _register_dummy_plugin(registry, plugin_id="dummy_b")

    with pytest.raises(ValueError, match="pass --env"):
        _resolve_initial_plugin_id(registry, None)
