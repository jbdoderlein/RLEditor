from __future__ import annotations

from rleditor.core.models import (
    AgentSpec,
    Breakpoint,
    BreakpointEvent,
    Checkpoint,
    DerivedTaskDefinition,
    EpisodeMoment,
    EpisodeStep,
    EpisodeTrace,
    RunConfig,
    TaskDefinition,
    TaskSnapshot,
    TrainingRun,
    TrainingStatus,
)


class _FakeArray:
    def tolist(self):
        return [1, 2, 3]


def test_task_definition_roundtrip_supports_reward_and_termination_config() -> None:
    task = TaskDefinition(
        environment_id="frozen_lake",
        name="Main Task",
        task_id="task_main",
        config={"grid": _FakeArray()},
        reward_config={"tile:G": 1.0},
        termination_config={"max_steps": 32},
        metadata={"experiment": "exp_01"},
    )

    payload = task.to_dict()
    assert payload["config"]["grid"] == [1, 2, 3]
    assert payload["reward_config"]["tile:G"] == 1.0

    restored = TaskDefinition.from_dict(payload)
    assert restored.environment_id == "frozen_lake"
    assert restored.task_id == "task_main"
    assert restored.reward_config == {"tile:G": 1.0}
    assert restored.termination_config == {"max_steps": 32}
    assert restored.metadata == {"experiment": "exp_01"}


def test_derived_task_roundtrip_preserves_lineage_fields() -> None:
    derived_task = DerivedTaskDefinition(
        environment_id="frozen_lake",
        name="Goal Corridor",
        parent_task_id="task_main",
        derivation_reason="focus_failure_zone",
        source_episode_id=7,
        source_moment_index=26,
        source_run_id="run_alpha",
    )

    payload = derived_task.to_dict()
    restored = DerivedTaskDefinition.from_dict(payload)

    assert restored.parent_task_id == "task_main"
    assert restored.derivation_reason == "focus_failure_zone"
    assert restored.source_episode_id == 7
    assert restored.source_moment_index == 26
    assert restored.source_run_id == "run_alpha"


def test_run_config_roundtrip_preserves_breakpoint_actions() -> None:
    config = RunConfig(
        run_config_id="run_cfg_1",
        algorithm="ppo",
        seed=1337,
        episode_trace_sample_rate=0.25,
        max_steps=2000,
        max_episodes=75,
        max_steps_per_episode=64,
        max_duration_seconds=120.0,
        breakpoints=[
            Breakpoint(kind="success_rate_gte", value=0.95, actions=["pause", "checkpoint"]),
            Breakpoint(kind="max_step", value=500.0, actions=["stop"]),
        ],
    )

    payload = config.to_dict()
    restored = RunConfig.from_dict(payload)

    assert restored.run_config_id == "run_cfg_1"
    assert restored.seed == 1337
    assert restored.episode_trace_sample_rate == 0.25
    assert restored.max_episodes == 75
    assert restored.max_steps_per_episode == 64
    assert restored.max_duration_seconds == 120.0
    assert restored.max_steps == 2000
    assert [rule.kind for rule in restored.breakpoints] == ["success_rate_gte", "max_step"]
    assert restored.breakpoints[0].actions == ["pause", "checkpoint"]
    assert restored.breakpoints[1].actions == ["stop"]


def test_run_config_supports_unlimited_max_steps() -> None:
    config = RunConfig(max_steps=-1)

    payload = config.to_dict()
    restored = RunConfig.from_dict(payload)

    assert config.max_steps is None
    assert payload["max_steps"] is None
    assert restored.max_steps is None


def test_episode_trace_roundtrip_preserves_task_snapshot_and_initial_observation() -> None:
    trace = EpisodeTrace(
        episode_id=3,
        total_reward=1.0,
        success=True,
        run_id="run_alpha",
        steps=[
            EpisodeStep(
                t=0,
                observation=0,
                action=1,
                reward=0.0,
                next_observation=1,
                terminated=False,
            ),
            EpisodeStep(
                t=1,
                observation=1,
                action=2,
                reward=1.0,
                next_observation=15,
                terminated=True,
            ),
        ],
        moments=[
            EpisodeMoment(
                episode_id=3,
                moment_index=0,
                observation=0,
                restorable_env_state={"state": 0},
            ),
            EpisodeMoment(
                episode_id=3,
                moment_index=1,
                observation=1,
                action_taken=1,
                reward=0.0,
                restorable_env_state={"state": 1},
            ),
            EpisodeMoment(
                episode_id=3,
                moment_index=2,
                observation=15,
                action_taken=2,
                reward=1.0,
                restorable_env_state={"state": 15},
            ),
        ],
        task_snapshot=TaskSnapshot(
            environment_id="frozen_lake",
            task_name="Main",
            task_id="task_main",
            task_config={"size": 4},
            reward_config={"tile:G": 1.0},
            termination_config={"max_steps": 64},
        ),
        initial_observation=0,
        metadata={"source": "unit_test"},
    )

    payload = trace.to_dict()
    restored = EpisodeTrace.from_dict(payload)

    assert restored.run_id == "run_alpha"
    assert restored.task_snapshot is not None
    assert restored.task_snapshot.task_id == "task_main"
    assert restored.task_snapshot.reward_config["tile:G"] == 1.0
    assert restored.initial_observation == 0
    assert len(restored.steps) == 2
    assert len(restored.moments) == 3
    assert restored.moments[2].restorable_env_state == {"state": 15}
    assert restored.steps[1].terminated is True


def test_episode_moment_roundtrip_preserves_restorable_state() -> None:
    moment = EpisodeMoment(
        episode_id=4,
        moment_index=9,
        observation={"state": 9},
        action_taken=2,
        reward=0.5,
        restorable_env_state={"opaque": "blob"},
    )

    restored = EpisodeMoment.from_dict(moment.to_dict())
    assert restored.episode_id == 4
    assert restored.moment_index == 9
    assert restored.restorable_env_state == {"opaque": "blob"}


def test_episode_trace_roundtrip_preserves_vector_actions() -> None:
    trace = EpisodeTrace(
        episode_id=7,
        total_reward=-1.0,
        success=False,
        steps=[
            EpisodeStep(
                t=0,
                observation=[0.1, 0.2],
                action=[-0.5, 0.25],
                reward=-1.0,
                next_observation=[0.0, 0.3],
                terminated=False,
                truncated=True,
            )
        ],
        moments=[
            EpisodeMoment(
                episode_id=7,
                moment_index=1,
                observation=[0.0, 0.3],
                action_taken=[-0.5, 0.25],
                reward=-1.0,
            )
        ],
    )

    restored = EpisodeTrace.from_dict(trace.to_dict())

    assert restored.steps[0].action == [-0.5, 0.25]
    assert restored.moments[0].action_taken == [-0.5, 0.25]


def test_checkpoint_roundtrip_preserves_parent_lineage() -> None:
    checkpoint = Checkpoint(
        checkpoint_id="checkpoint_002",
        label="Checkpoint 002",
        created_at="2026-03-30 10:00:00",
        reason="run_finished",
        parent_checkpoint_id="checkpoint_001",
    )

    restored = Checkpoint.from_dict(checkpoint.to_dict())
    assert restored.checkpoint_id == "checkpoint_002"
    assert restored.parent_checkpoint_id == "checkpoint_001"


def test_agent_spec_roundtrip_exposes_default_hyperparameters() -> None:
    agent_spec = AgentSpec(
        agent_spec_id="agent_spec_a",
        algorithm="dqn",
        policy="mlp",
        default_hyperparameters={"lr": 0.001},
        seed=42,
    )

    restored = AgentSpec.from_dict(agent_spec.to_dict())
    assert restored.agent_spec_id == "agent_spec_a"
    assert restored.policy == "mlp"
    assert restored.default_hyperparameters["lr"] == 0.001


def test_training_run_from_dict_falls_back_to_idle_on_invalid_status() -> None:
    run = TrainingRun.from_dict(
        {
            "run_id": "run_1",
            "status": "unknown",
            "agent_spec_id": "agent_spec_1",
            "started_at": "2026-04-07 12:00:00",
        }
    )
    assert run.status == TrainingStatus.IDLE
    assert run.agent_spec_id == "agent_spec_1"
    assert run.started_at == "2026-04-07 12:00:00"


def test_breakpoint_event_roundtrip_exposes_breakpoint() -> None:
    event = BreakpointEvent(
        breakpoint=Breakpoint(kind="mean_reward_gte", value=0.8, window=100, actions=["notify"]),
        step=1200,
        episode=44,
        message="hit",
    )

    restored = BreakpointEvent.from_dict(event.to_dict())
    assert restored.breakpoint.kind == "mean_reward_gte"
    assert restored.breakpoint.window == 100
    assert restored.breakpoint.actions == ["notify"]
    assert restored.step == 1200
