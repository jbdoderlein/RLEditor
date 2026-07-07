from __future__ import annotations

from rleditor.plugins.base import EnvironmentPlugin


class PluginRegistry:
    """Registry for all extension points (environments and related GUI providers)."""

    def __init__(self) -> None:
        self._env_plugins: dict[str, EnvironmentPlugin] = {}

    def register_environment(self, plugin: EnvironmentPlugin) -> None:
        if plugin.plugin_id in self._env_plugins:
            msg = f"Environment plugin '{plugin.plugin_id}' is already registered"
            raise ValueError(msg)
        self._env_plugins[plugin.plugin_id] = plugin

    def list_environment_plugins(self) -> list[EnvironmentPlugin]:
        return sorted(self._env_plugins.values(), key=lambda plugin: plugin.display_name)

    def get_environment_plugin(self, plugin_id: str) -> EnvironmentPlugin:
        if plugin_id not in self._env_plugins:
            msg = f"Unknown environment plugin: {plugin_id}"
            raise KeyError(msg)
        return self._env_plugins[plugin_id]


def register_builtin_plugins(registry: PluginRegistry) -> None:
    from rleditor.plugins.builtin.blackjack import build_blackjack_plugin
    from rleditor.plugins.builtin.frozen_lake import build_frozen_lake_plugin
    from rleditor.plugins.builtin.mujoco import build_mujoco_plugin

    registry.register_environment(build_blackjack_plugin())
    registry.register_environment(build_frozen_lake_plugin())
    registry.register_environment(build_mujoco_plugin())
