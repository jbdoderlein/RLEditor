from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from PySide6.QtWidgets import QWidget

from rleditor.core.models import EpisodeTrace, TaskDefinition, TaskDerivationOptions


class EpisodeReplayWidget(QWidget):
    """Widget contract for plugin-specific episode step visualization."""

    def set_frame(
        self,
        trace: EpisodeTrace,
        step_index: int,
    ) -> None:
        raise NotImplementedError


class RestorableEnvironment(Protocol):
    """Gym-compatible environment extended with state transfer hooks."""

    action_space: Any
    observation_space: Any

    def reset(self, *, seed: int | None = None) -> Any:
        ...

    def step(self, action: Any) -> Any:
        ...

    def render(self) -> Any:
        ...

    def close(self) -> None:
        ...

    def export_state(self) -> Any:
        ...

    def import_state(self, state: Any) -> Any:
        ...

    def reinstantiate(self, *, render_mode: str | None = None) -> RestorableEnvironment:
        ...


class EnvironmentBackend(Protocol):
    """Backend API to build and run environments for a plugin."""

    def default_task(self) -> TaskDefinition:
        ...

    def create_env(self, task: TaskDefinition) -> RestorableEnvironment:
        ...

    def derive_task_from_episode(
        self,
        source_task: TaskDefinition,
        trace: EpisodeTrace,
        moment_index: int,
    ) -> TaskDerivationOptions | None:
        ...


class EnvironmentGuiExtension(Protocol):
    """GUI hooks for plugin-specific task editing and state rendering."""

    def create_task_editor_widget(
        self,
        task: TaskDefinition,
        on_task_changed: Callable[[TaskDefinition], None],
    ) -> QWidget:
        ...

    def create_episode_replay_widget(self, parent: QWidget | None = None) -> EpisodeReplayWidget | None:
        ...


@dataclass(slots=True)
class EnvironmentPlugin:
    plugin_id: str
    display_name: str
    description: str
    backend: EnvironmentBackend
    gui_extension: EnvironmentGuiExtension | None = None
