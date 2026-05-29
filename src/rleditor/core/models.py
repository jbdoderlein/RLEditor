from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _to_serializable(value.to_dict())
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_serializable(value.tolist())
    return repr(value)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_actions(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        normalized = [str(item) for item in value if str(item)]
        if normalized:
            return normalized
    return ["pause"]


class TrainingStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FINISHED = "finished"


@dataclass(slots=True)
class EnvironmentCapabilities:
    """Capability declaration for one environment adapter."""

    create_environment: bool = True
    serialize_task: bool = True
    record_episode_trace: bool = True
    render_episode_moment: bool = True
    export_restorable_state: bool = False
    import_restorable_state: bool = False
    validate_state_compatibility: bool = False
    mutate_task: bool = False
    derive_task_from_episode: bool = False
    compare_episode_moments: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "create_environment": self.create_environment,
            "serialize_task": self.serialize_task,
            "record_episode_trace": self.record_episode_trace,
            "render_episode_moment": self.render_episode_moment,
            "export_restorable_state": self.export_restorable_state,
            "import_restorable_state": self.import_restorable_state,
            "validate_state_compatibility": self.validate_state_compatibility,
            "mutate_task": self.mutate_task,
            "derive_task_from_episode": self.derive_task_from_episode,
            "compare_episode_moments": self.compare_episode_moments,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EnvironmentCapabilities:
        return cls(
            create_environment=bool(payload.get("create_environment", True)),
            serialize_task=bool(payload.get("serialize_task", True)),
            record_episode_trace=bool(payload.get("record_episode_trace", True)),
            render_episode_moment=bool(payload.get("render_episode_moment", True)),
            export_restorable_state=bool(payload.get("export_restorable_state", False)),
            import_restorable_state=bool(payload.get("import_restorable_state", False)),
            validate_state_compatibility=bool(payload.get("validate_state_compatibility", False)),
            mutate_task=bool(payload.get("mutate_task", False)),
            derive_task_from_episode=bool(payload.get("derive_task_from_episode", False)),
            compare_episode_moments=bool(payload.get("compare_episode_moments", False)),
        )


@dataclass(slots=True)
class Breakpoint:
    """A training breakpoint condition evaluated during learning."""

    kind: str
    value: float
    window: int | None = None
    breakpoint_id: str | None = None
    actions: list[str] = field(default_factory=lambda: ["pause"])
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "breakpoint_id": self.breakpoint_id,
            "label": self.label,
            "kind": self.kind,
            "value": self.value,
            "window": self.window,
            "actions": list(self.actions),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Breakpoint:
        window = payload.get("window")
        return cls(
            kind=str(payload.get("kind", "")),
            value=float(payload.get("value", 0.0)),
            window=int(window) if window is not None else None,
            breakpoint_id=payload.get("breakpoint_id"),
            actions=_as_actions(payload.get("actions")),
            label=payload.get("label"),
        )


@dataclass(slots=True)
class TaskDefinition:
    """Serializable task definition for an environment plugin."""

    environment_id: str
    name: str
    task_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    reward_config: dict[str, float] = field(default_factory=dict)
    termination_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "name": self.name,
            "task_id": self.task_id,
            "config": _to_serializable(self.config),
            "reward_config": _to_serializable(self.reward_config),
            "termination_config": _to_serializable(self.termination_config),
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskDefinition:
        return cls(
            environment_id=str(payload.get("environment_id", "")),
            name=str(payload.get("name", "")),
            task_id=payload.get("task_id"),
            config=_as_dict(payload.get("config")),
            reward_config=_as_dict(payload.get("reward_config")),
            termination_config=_as_dict(payload.get("termination_config")),
            metadata=_as_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class DerivedTaskDefinition(TaskDefinition):
    """A task derived from an existing parent task context."""

    derived_task_id: str | None = None
    parent_task_id: str | None = None
    derivation_reason: str | None = None
    source_episode_id: int | None = None
    source_moment_index: int | None = None
    source_run_id: str | None = None
    start_state: Any | None = None
    goal_state: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = TaskDefinition.to_dict(self)
        payload.update(
            {
                "derived_task_id": self.derived_task_id,
                "parent_task_id": self.parent_task_id,
                "derivation_reason": self.derivation_reason,
                "source_episode_id": self.source_episode_id,
                "source_moment_index": self.source_moment_index,
                "source_run_id": self.source_run_id,
                "start_state": _to_serializable(self.start_state),
                "goal_state": _to_serializable(self.goal_state),
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DerivedTaskDefinition:
        return cls(
            environment_id=str(payload.get("environment_id", "")),
            name=str(payload.get("name", "")),
            task_id=payload.get("task_id"),
            config=_as_dict(payload.get("config")),
            reward_config=_as_dict(payload.get("reward_config")),
            termination_config=_as_dict(payload.get("termination_config")),
            metadata=_as_dict(payload.get("metadata")),
            derived_task_id=payload.get("derived_task_id"),
            parent_task_id=payload.get("parent_task_id"),
            derivation_reason=payload.get("derivation_reason"),
            source_episode_id=(
                int(payload.get("source_episode_id"))
                if payload.get("source_episode_id") is not None
                else None
            ),
            source_moment_index=(
                int(payload.get("source_moment_index"))
                if payload.get("source_moment_index") is not None
                else None
            ),
            source_run_id=payload.get("source_run_id"),
            start_state=payload.get("start_state"),
            goal_state=payload.get("goal_state"),
        )


@dataclass(slots=True)
class TaskDerivationOptions:
    """Optional plugin-provided settings applied when deriving a task."""

    config_updates: dict[str, Any] = field(default_factory=dict)
    reward_config_updates: dict[str, float] = field(default_factory=dict)
    termination_config_updates: dict[str, Any] = field(default_factory=dict)
    derivation_reason: str | None = None
    source_episode_id: int | None = None
    source_moment_index: int | None = None
    source_run_id: str | None = None
    start_state: Any | None = None
    goal_state: Any | None = None


@dataclass(slots=True)
class RunConfig:
    run_config_id: str | None = None
    algorithm: str = "q_learning"
    seed: int | None = None
    episode_trace_sample_rate: float = 1.0
    max_steps: int | None = 10_000
    max_episodes: int | None = None
    max_steps_per_episode: int | None = None
    max_duration_seconds: float | None = None
    learning_rate: float = 0.1
    gamma: float = 0.99
    epsilon: float = 1.0
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    breakpoints: list[Breakpoint] = field(default_factory=list)
    checkpoint_policy: dict[str, Any] = field(default_factory=dict)
    evaluation_policy: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.episode_trace_sample_rate = max(0.0, min(1.0, float(self.episode_trace_sample_rate)))
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)
            if self.max_steps <= 0:
                self.max_steps = None
        if not self.hyperparameters:
            self.hyperparameters = {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
            }
        else:
            self.learning_rate = float(
                self.hyperparameters.get("learning_rate", self.hyperparameters.get("lr", self.learning_rate))
            )
            self.gamma = float(self.hyperparameters.get("gamma", self.gamma))
            self.epsilon = float(self.hyperparameters.get("epsilon", self.epsilon))

    def to_dict(self) -> dict[str, Any]:
        hyperparameters = dict(self.hyperparameters)
        hyperparameters.setdefault("learning_rate", self.learning_rate)
        hyperparameters.setdefault("gamma", self.gamma)
        hyperparameters.setdefault("epsilon", self.epsilon)

        return {
            "run_config_id": self.run_config_id,
            "algorithm": self.algorithm,
            "seed": self.seed,
            "episode_trace_sample_rate": self.episode_trace_sample_rate,
            "max_steps": self.max_steps,
            "max_episodes": self.max_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "max_duration_seconds": self.max_duration_seconds,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "hyperparameters": _to_serializable(hyperparameters),
            "breakpoints": [breakpoint.to_dict() for breakpoint in self.breakpoints],
            "checkpoint_policy": _to_serializable(self.checkpoint_policy),
            "evaluation_policy": _to_serializable(self.evaluation_policy),
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunConfig:
        breakpoints_payload = payload.get("breakpoints", [])
        breakpoints = [
            Breakpoint.from_dict(item)
            for item in breakpoints_payload
            if isinstance(item, dict)
        ]
        max_steps_payload = payload.get("max_steps", 10_000)
        return cls(
            run_config_id=payload.get("run_config_id"),
            algorithm=str(payload.get("algorithm", "q_learning")),
            seed=int(payload.get("seed")) if payload.get("seed") is not None else None,
            episode_trace_sample_rate=float(payload.get("episode_trace_sample_rate", 1.0)),
            max_steps=(
                int(max_steps_payload)
                if max_steps_payload is not None
                else None
            ),
            max_episodes=(
                int(payload.get("max_episodes"))
                if payload.get("max_episodes") is not None
                else None
            ),
            max_steps_per_episode=(
                int(payload.get("max_steps_per_episode"))
                if payload.get("max_steps_per_episode") is not None
                else None
            ),
            max_duration_seconds=(
                float(payload.get("max_duration_seconds"))
                if payload.get("max_duration_seconds") is not None
                else None
            ),
            learning_rate=float(payload.get("learning_rate", 0.1)),
            gamma=float(payload.get("gamma", 0.99)),
            epsilon=float(payload.get("epsilon", 1.0)),
            hyperparameters=_as_dict(payload.get("hyperparameters")),
            breakpoints=breakpoints,
            checkpoint_policy=_as_dict(payload.get("checkpoint_policy")),
            evaluation_policy=_as_dict(payload.get("evaluation_policy")),
            metadata=_as_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class TrainingMetrics:
    step: int = 0
    episode: int = 0
    reward_step: float = 0.0
    cumulative_reward: float = 0.0
    mean_reward: float = 0.0
    episode_reward_mean: float = 0.0
    success_rate: float = 0.0
    episode_length_mean: float = 0.0
    fps: float = 0.0
    exploration_rate: float = 0.0
    value_loss: float | None = None
    policy_loss: float | None = None


@dataclass(slots=True)
class EpisodeStep:
    t: int
    observation: Any
    action: Any
    reward: float
    next_observation: Any
    terminated: bool
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "observation": _to_serializable(self.observation),
            "action": _to_serializable(self.action),
            "reward": self.reward,
            "next_observation": _to_serializable(self.next_observation),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": _to_serializable(self.info),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodeStep:
        return cls(
            t=int(payload.get("t", 0)),
            observation=payload.get("observation"),
            action=payload.get("action", 0),
            reward=float(payload.get("reward", 0.0)),
            next_observation=payload.get("next_observation"),
            terminated=bool(payload.get("terminated", False)),
            truncated=bool(payload.get("truncated", False)),
            info=_as_dict(payload.get("info")),
        )


@dataclass(slots=True)
class TaskSnapshot:
    environment_id: str
    task_name: str
    task_id: str | None = None
    task_config: dict[str, Any] = field(default_factory=dict)
    reward_config: dict[str, float] = field(default_factory=dict)
    termination_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "task_name": self.task_name,
            "task_id": self.task_id,
            "task_config": _to_serializable(self.task_config),
            "reward_config": _to_serializable(self.reward_config),
            "termination_config": _to_serializable(self.termination_config),
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskSnapshot:
        return cls(
            environment_id=str(payload.get("environment_id", "")),
            task_name=str(payload.get("task_name", "")),
            task_id=payload.get("task_id"),
            task_config=_as_dict(payload.get("task_config")),
            reward_config=_as_dict(payload.get("reward_config")),
            termination_config=_as_dict(payload.get("termination_config")),
            metadata=_as_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class EpisodeMoment:
    episode_id: int
    moment_index: int
    observation: Any = None
    action_taken: Any | None = None
    reward: float | None = None
    restorable_env_state: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "moment_index": self.moment_index,
            "observation": _to_serializable(self.observation),
            "action_taken": _to_serializable(self.action_taken),
            "reward": self.reward,
            "restorable_env_state": _to_serializable(self.restorable_env_state),
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodeMoment:
        return cls(
            episode_id=int(payload.get("episode_id", 0)),
            moment_index=int(payload.get("moment_index", 0)),
            observation=payload.get("observation"),
            action_taken=payload.get("action_taken"),
            reward=(
                float(payload.get("reward"))
                if payload.get("reward") is not None
                else None
            ),
            restorable_env_state=payload.get("restorable_env_state"),
            metadata=_as_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class EpisodeTrace:
    episode_id: int
    total_reward: float
    success: bool
    run_id: str | None = None
    steps: list[EpisodeStep] = field(default_factory=list)
    moments: list[EpisodeMoment] = field(default_factory=list)
    task_snapshot: TaskSnapshot | None = None
    initial_observation: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "run_id": self.run_id,
            "total_reward": self.total_reward,
            "success": self.success,
            "initial_observation": _to_serializable(self.initial_observation),
            "steps": [step.to_dict() for step in self.steps],
            "moments": [moment.to_dict() for moment in self.moments],
            "task_snapshot": None if self.task_snapshot is None else self.task_snapshot.to_dict(),
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodeTrace:
        snapshot_payload = payload.get("task_snapshot")
        task_snapshot = None
        if isinstance(snapshot_payload, dict):
            task_snapshot = TaskSnapshot.from_dict(snapshot_payload)

        step_payloads = payload.get("steps", [])
        steps = [
            EpisodeStep.from_dict(step_payload)
            for step_payload in step_payloads
            if isinstance(step_payload, dict)
        ]
        moment_payloads = payload.get("moments", [])
        moments = [
            EpisodeMoment.from_dict(moment_payload)
            for moment_payload in moment_payloads
            if isinstance(moment_payload, dict)
        ]

        return cls(
            episode_id=int(payload.get("episode_id", 0)),
            run_id=payload.get("run_id"),
            total_reward=float(payload.get("total_reward", 0.0)),
            success=bool(payload.get("success", False)),
            steps=steps,
            moments=moments,
            task_snapshot=task_snapshot,
            initial_observation=payload.get("initial_observation"),
            metadata=_as_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class BreakpointEvent:
    breakpoint: Breakpoint
    step: int
    episode: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "breakpoint": self.breakpoint.to_dict(),
            "step": self.step,
            "episode": self.episode,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BreakpointEvent:
        breakpoint_payload = payload.get("breakpoint", {})
        if not isinstance(breakpoint_payload, dict):
            breakpoint_payload = {}
        return cls(
            breakpoint=Breakpoint.from_dict(breakpoint_payload),
            step=int(payload.get("step", 0)),
            episode=int(payload.get("episode", 0)),
            message=str(payload.get("message", "")),
        )


@dataclass(slots=True)
class AgentSpec:
    """Declarative agent configuration used to initialize training runs."""

    agent_spec_id: str
    algorithm: str
    policy: str
    default_hyperparameters: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_spec_id": self.agent_spec_id,
            "algorithm": self.algorithm,
            "policy": self.policy,
            "default_hyperparameters": _to_serializable(self.default_hyperparameters),
            "seed": self.seed,
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AgentSpec:
        return cls(
            agent_spec_id=str(payload.get("agent_spec_id", "")),
            algorithm=str(payload.get("algorithm", "")),
            policy=str(payload.get("policy", "")),
            default_hyperparameters=_as_dict(payload.get("default_hyperparameters")),
            seed=int(payload.get("seed")) if payload.get("seed") is not None else None,
            metadata=_as_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class TrainingRun:
    """Execution context linking one agent specification, one task and one run."""

    run_id: str
    task_id: str | None = None
    agent_spec_id: str | None = None
    run_config_id: str | None = None
    status: TrainingStatus = TrainingStatus.IDLE
    started_at: str | None = None
    ended_at: str | None = None
    parent_checkpoint_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent_spec_id": self.agent_spec_id,
            "run_config_id": self.run_config_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingRun:
        raw_status = str(payload.get("status", TrainingStatus.IDLE.value))
        try:
            status = TrainingStatus(raw_status)
        except ValueError:
            status = TrainingStatus.IDLE

        return cls(
            run_id=str(payload.get("run_id", "")),
            task_id=payload.get("task_id"),
            agent_spec_id=payload.get("agent_spec_id"),
            run_config_id=payload.get("run_config_id"),
            status=status,
            started_at=payload.get("started_at"),
            ended_at=payload.get("ended_at"),
            parent_checkpoint_id=payload.get("parent_checkpoint_id"),
            metadata=_as_dict(payload.get("metadata")),
        )


@dataclass(slots=True)
class Checkpoint:
    checkpoint_id: str
    label: str
    created_at: str
    reason: str
    parent_checkpoint_id: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    task_name: str | None = None
    storage_uri: str | None = None
    step: int = 0
    episode: int = 0
    task_snapshot: TaskSnapshot | None = None
    agent_spec_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "label": self.label,
            "created_at": self.created_at,
            "reason": self.reason,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "storage_uri": self.storage_uri,
            "step": self.step,
            "episode": self.episode,
            "task_snapshot": None if self.task_snapshot is None else self.task_snapshot.to_dict(),
            "agent_spec_id": self.agent_spec_id,
            "metadata": _to_serializable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Checkpoint:
        task_snapshot_payload = payload.get("task_snapshot")
        task_snapshot = None
        if isinstance(task_snapshot_payload, dict):
            task_snapshot = TaskSnapshot.from_dict(task_snapshot_payload)

        return cls(
            checkpoint_id=str(payload.get("checkpoint_id", "")),
            label=str(payload.get("label", "")),
            created_at=str(payload.get("created_at", "")),
            reason=str(payload.get("reason", "")),
            parent_checkpoint_id=payload.get("parent_checkpoint_id"),
            run_id=payload.get("run_id"),
            task_id=payload.get("task_id"),
            task_name=payload.get("task_name"),
            storage_uri=payload.get("storage_uri"),
            step=int(payload.get("step", 0)),
            episode=int(payload.get("episode", 0)),
            task_snapshot=task_snapshot,
            agent_spec_id=payload.get("agent_spec_id"),
            metadata=_as_dict(payload.get("metadata")),
        )


__all__ = [
    "AgentSpec",
    "Breakpoint",
    "BreakpointEvent",
    "Checkpoint",
    "DerivedTaskDefinition",
    "EnvironmentCapabilities",
    "EpisodeMoment",
    "EpisodeStep",
    "EpisodeTrace",
    "RunConfig",
    "TaskDefinition",
    "TaskDerivationOptions",
    "TaskSnapshot",
    "TrainingMetrics",
    "TrainingRun",
    "TrainingStatus",
]
