from __future__ import annotations

from collections import deque
from collections.abc import Callable
import random
import threading
import time
from typing import Any

from PySide6.QtCore import QObject, Signal

from rleditor.core.models import (
    Breakpoint,
    BreakpointEvent,
    Checkpoint,
    EpisodeMoment,
    EpisodeStep,
    EpisodeTrace,
    RunConfig,
    TaskDefinition,
    TaskSnapshot,
    TrainingMetrics,
    TrainingStatus,
)
from rleditor.infra.stable_baselines_backend import (
    StableBaselines3StepCallback,
    StableBaselines3TraceWrapper,
    create_stable_baselines3_model,
    export_stable_baselines3_learner_state,
    is_stable_baselines3_algorithm,
)


class TrainingRunner(QObject):
    """Training runner for explicit Gymnasium-compatible environment adapters."""

    status_changed = Signal(object)
    metrics_updated = Signal(object)
    episode_captured = Signal(object)
    breakpoint_triggered = Signal(object)

    _MAX_STEPS_PER_TIMER_TICK = 2_048
    _TIMER_TICK_BUDGET_SECONDS = 0.004
    _METRICS_EMIT_INTERVAL_SECONDS = 0.05
    _WORKER_YIELD_SECONDS = 0.001

    def __init__(self) -> None:
        super().__init__()
        self._status = TrainingStatus.IDLE
        self._run_id: str | None = None
        self._task: TaskDefinition | None = None
        self._config = RunConfig()
        self._random = random.Random()
        self._trace_random = random.Random()
        self._metrics = TrainingMetrics()
        self._started_at = 0.0
        self._last_tick_at = 0.0
        self._episode_rewards: deque[float] = deque(maxlen=100)
        self._episode_lengths: deque[int] = deque(maxlen=100)
        self._episode_successes: deque[int] = deque(maxlen=100)
        self._triggered_breakpoints: set[int] = set()
        self._env: Any | None = None
        self._env_factory: Callable[[TaskDefinition], Any] | None = None
        self._observation: Any | None = None
        self._episode_steps_buffer: list[EpisodeStep] = []
        self._episode_moments_buffer: list[EpisodeMoment] = []
        self._episode_reward_total = 0.0
        self._episode_step_counter = 0
        self._episode_seed: int | None = None
        self._record_current_episode_trace = False
        self._q_values: dict[tuple[str, int], float] = {}
        self._sb3_model: Any | None = None
        self._sb3_trace_env: StableBaselines3TraceWrapper | None = None
        self._pending_sb3_learner_state: dict[str, Any] | None = None
        self._sb3_pending_finish_status: TrainingStatus | None = None
        self._last_metrics_emitted_at = 0.0
        self._auto_run = False
        self._stop_requested = threading.Event()
        self._resume_requested = threading.Event()
        self._resume_requested.set()
        self._worker_thread: threading.Thread | None = None

    @property
    def status(self) -> TrainingStatus:
        return self._status

    def start(
        self,
        task: TaskDefinition,
        config: RunConfig,
        *,
        run_id: str | None = None,
        env_factory: Callable[[TaskDefinition], Any] | None = None,
        initial_checkpoint: Checkpoint | None = None,
        auto_run: bool = False,
    ) -> None:
        self._run_id = run_id
        self._task = task
        self._config = config
        self._env_factory = env_factory
        self._auto_run = auto_run
        self._env = None
        self._observation = None
        self._episode_steps_buffer = []
        self._episode_reward_total = 0.0
        self._episode_step_counter = 0
        self._episode_seed = None
        self._record_current_episode_trace = False
        self._q_values = {}
        self._sb3_model = None
        self._sb3_trace_env = None
        self._pending_sb3_learner_state = None
        self._sb3_pending_finish_status = None
        self._restore_checkpoint_state(initial_checkpoint)
        if config.seed is not None:
            self._random.seed(config.seed)
            self._trace_random.seed(config.seed + 1_000_003)
        else:
            self._random.seed()
            self._trace_random.seed()
        self._metrics = TrainingMetrics()
        self._metrics.exploration_rate = config.epsilon
        self._episode_rewards.clear()
        self._episode_lengths.clear()
        self._episode_successes.clear()
        self._triggered_breakpoints.clear()
        self._started_at = time.perf_counter()
        self._last_tick_at = self._started_at
        self._last_metrics_emitted_at = self._started_at
        self._stop_requested.clear()
        self._resume_requested.set()

        self._connect_environment_or_raise()
        self._set_status(TrainingStatus.RUNNING)
        if self._auto_run:
            self._start_worker_thread()

    def start_background(self) -> None:
        if self._status != TrainingStatus.RUNNING:
            return
        self._auto_run = True
        self._start_worker_thread()

    def pause(self) -> None:
        if self._status == TrainingStatus.RUNNING:
            self._set_status(TrainingStatus.PAUSED)

    def resume(self) -> None:
        if self._status == TrainingStatus.PAUSED:
            self._set_status(TrainingStatus.RUNNING)

    def stop(self) -> None:
        if not self._auto_run:
            self._finish_run(TrainingStatus.STOPPED)
            return
        self._stop_requested.set()
        self._resume_requested.set()

    def _set_status(self, status: TrainingStatus) -> None:
        self._status = status
        if status == TrainingStatus.RUNNING:
            self._last_tick_at = time.perf_counter()
            self._resume_requested.set()
        elif status == TrainingStatus.PAUSED:
            self._resume_requested.clear()
        else:
            self._resume_requested.set()
        self.status_changed.emit(status)

    def _start_worker_thread(self) -> None:
        if self._worker_thread is not None and not self._worker_thread.is_alive():
            self._worker_thread = None
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(
            target=self._run_loop,
            name="TrainingRunnerLoop",
            daemon=True,
        )
        self._worker_thread.start()

    def _run_loop(self) -> None:
        while True:
            if self._stop_requested.is_set():
                if self._status not in {TrainingStatus.FINISHED, TrainingStatus.STOPPED}:
                    self._finish_run(TrainingStatus.STOPPED)
                break

            if self._status == TrainingStatus.PAUSED:
                self._resume_requested.wait(timeout=0.05)
                continue

            if self._status != TrainingStatus.RUNNING:
                break

            batch_started_at = time.perf_counter()
            steps_processed = 0
            while (
                self._status == TrainingStatus.RUNNING
                and not self._stop_requested.is_set()
                and steps_processed < self._MAX_STEPS_PER_TIMER_TICK
            ):
                if not self._on_tick(emit_metrics=False):
                    break
                steps_processed += 1
                if (time.perf_counter() - batch_started_at) >= self._TIMER_TICK_BUDGET_SECONDS:
                    break

            if steps_processed <= 0:
                time.sleep(0.001)
                continue

            now = time.perf_counter()
            if (
                (now - self._last_metrics_emitted_at) >= self._METRICS_EMIT_INTERVAL_SECONDS
                or self._status != TrainingStatus.RUNNING
            ):
                self._emit_metrics_updated(now=now)
            time.sleep(self._WORKER_YIELD_SECONDS)

    def _on_tick(self, *, emit_metrics: bool = True) -> bool:
        if self._status != TrainingStatus.RUNNING:
            return False

        if self._env is None:
            msg = "Training environment is not available; cannot step runner."
            raise RuntimeError(msg)

        if is_stable_baselines3_algorithm(self._config.algorithm):
            self._on_tick_stable_baselines3(emit_metrics=emit_metrics)
            return True

        self._on_tick_gym(emit_metrics=emit_metrics)
        return True

    def _on_tick_stable_baselines3(self, *, emit_metrics: bool = True) -> None:
        if self._sb3_model is None or self._sb3_trace_env is None:
            msg = "Stable-Baselines3 model is not available; cannot step runner."
            raise RuntimeError(msg)

        max_steps = self._config.max_steps
        remaining_steps = None if max_steps is None else max_steps - self._metrics.step
        if remaining_steps is not None and remaining_steps <= 0:
            self._finish_run(TrainingStatus.FINISHED)
            return

        chunk_steps = self._stable_baselines3_chunk_steps()
        if remaining_steps is not None:
            chunk_steps = min(chunk_steps, remaining_steps)

        started_at = time.perf_counter()
        start_step = self._metrics.step
        callback = StableBaselines3StepCallback(self._on_stable_baselines3_step)
        self._sb3_model.learn(
            total_timesteps=chunk_steps,
            reset_num_timesteps=False,
            callback=callback.callback,
            progress_bar=False,
        )
        self._drain_stable_baselines3_episodes()

        elapsed = max(time.perf_counter() - started_at, 1e-6)
        processed_steps = max(0, self._metrics.step - start_step)
        if processed_steps > 0:
            self._metrics.fps = processed_steps / elapsed

        if emit_metrics:
            self._emit_metrics_updated()

        if self._sb3_pending_finish_status is not None:
            status = self._sb3_pending_finish_status
            self._sb3_pending_finish_status = None
            self._finish_run(status)

    def _on_stable_baselines3_step(self, model: Any) -> bool:
        if self._status != TrainingStatus.RUNNING or self._stop_requested.is_set():
            return False

        self._metrics.step += 1
        exploration_rate = getattr(model, "exploration_rate", None)
        if exploration_rate is not None:
            try:
                self._metrics.exploration_rate = float(exploration_rate)
            except (TypeError, ValueError):
                pass
        self._metrics.value_loss = None
        self._metrics.policy_loss = None

        self._drain_stable_baselines3_episodes()

        if self._config.max_steps is not None and self._metrics.step >= self._config.max_steps:
            self._sb3_pending_finish_status = TrainingStatus.FINISHED
            return False

        if (
            self._config.max_duration_seconds is not None
            and (time.perf_counter() - self._started_at) >= self._config.max_duration_seconds
        ):
            self._sb3_pending_finish_status = TrainingStatus.FINISHED
            return False

        if self._evaluate_breakpoints(defer_stop=True):
            return False
        return True

    def _drain_stable_baselines3_episodes(self) -> None:
        trace_env = self._sb3_trace_env
        if trace_env is None:
            return

        for summary in trace_env.drain_episode_summaries():
            self._metrics.episode += 1
            self._metrics.cumulative_reward += summary.total_reward
            self._episode_rewards.append(summary.total_reward)
            self._episode_lengths.append(summary.length)
            self._episode_successes.append(1 if summary.success else 0)

            self._metrics.episode_reward_mean = sum(self._episode_rewards) / len(self._episode_rewards)
            self._metrics.mean_reward = self._metrics.episode_reward_mean
            self._metrics.episode_length_mean = sum(self._episode_lengths) / len(self._episode_lengths)
            self._metrics.success_rate = sum(self._episode_successes) / len(self._episode_successes)

            if self._config.max_episodes is not None and self._metrics.episode >= self._config.max_episodes:
                self._sb3_pending_finish_status = TrainingStatus.FINISHED

        for trace in trace_env.drain_traces():
            self.episode_captured.emit(trace)

    def _stable_baselines3_chunk_steps(self) -> int:
        raw_value = self._config.hyperparameters.get("sb3_train_chunk_steps", 64)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = 64
        return max(1, value)

    def _on_tick_gym(self, *, emit_metrics: bool = True) -> None:
        env = self._env
        if env is None:
            msg = "Training environment is not available; cannot step runner."
            raise RuntimeError(msg)

        now = time.perf_counter()
        elapsed = max(now - self._last_tick_at, 1e-6)
        self._last_tick_at = now

        previous_observation = self._observation
        if previous_observation is None:
            self._reset_environment()
            previous_observation = self._observation
            if previous_observation is None:
                self._finish_run(TrainingStatus.STOPPED)
                return

        self._metrics.step += 1
        self._metrics.fps = 1.0 / elapsed

        if self._config.algorithm == "q_learning":
            self._metrics.exploration_rate = self._q_learning_exploration_rate()
        else:
            self._metrics.exploration_rate = 0.0

        action = self._select_action(previous_observation, env)

        step_result = env.step(action)
        if isinstance(step_result, tuple) and len(step_result) == 5:
            observation, reward, terminated, truncated, info = step_result
        elif isinstance(step_result, tuple) and len(step_result) == 4:
            observation, reward, done, info = step_result
            terminated = bool(done)
            truncated = False
        else:
            self._finish_run(TrainingStatus.STOPPED)
            return

        done = bool(terminated) or bool(truncated)
        reward_value = float(reward)
        info_payload = dict(info) if isinstance(info, dict) else {}
        normalized_previous_observation = self._normalize_observation(previous_observation)
        normalized_observation = self._normalize_observation(observation)

        max_steps_per_episode = self._config.max_steps_per_episode
        if (
            not done
            and max_steps_per_episode is not None
            and max_steps_per_episode > 0
            and (self._episode_step_counter + 1) >= max_steps_per_episode
        ):
            done = True
            info_payload["forced_failure"] = "max_steps_per_episode"
            info_payload["max_steps_per_episode"] = max_steps_per_episode

        if self._record_current_episode_trace:
            self._episode_steps_buffer.append(
                EpisodeStep(
                    t=self._episode_step_counter,
                    observation=normalized_previous_observation,
                    action=int(action),
                    reward=reward_value,
                    next_observation=normalized_observation,
                    terminated=done and info_payload.get("forced_failure") != "max_steps_per_episode",
                    truncated=bool(truncated) or info_payload.get("forced_failure") == "max_steps_per_episode",
                    info=info_payload,
                )
            )

            self._episode_moments_buffer.append(
                EpisodeMoment(
                    episode_id=self._current_episode_id(),
                    moment_index=self._episode_step_counter + 1,
                    observation=normalized_observation,
                    action_taken=int(action),
                    reward=reward_value,
                    restorable_env_state=self._maybe_export_restorable_state(env),
                    metadata={
                        "terminated": done and info_payload.get("forced_failure") != "max_steps_per_episode",
                        "truncated": bool(truncated)
                        or info_payload.get("forced_failure") == "max_steps_per_episode",
                        "info": info_payload,
                    },
                )
            )

        self._episode_step_counter += 1
        self._episode_reward_total += reward_value
        self._observation = normalized_observation

        self._metrics.reward_step = reward_value
        self._metrics.cumulative_reward += reward_value

        if self._config.algorithm == "q_learning" and not self._model_updates_disabled():
            self._update_q_learning(
                normalized_previous_observation,
                int(action),
                reward_value,
                normalized_observation,
                done,
                env,
            )
        else:
            if self._config.algorithm == "q_learning":
                self._metrics.value_loss = 0.0
                self._metrics.policy_loss = None
            else:
                self._metrics.value_loss = None
                self._metrics.policy_loss = None

        if done:
            self._metrics.episode += 1
            total_reward = self._episode_reward_total
            episode_length = self._episode_step_counter
            success = self._episode_success(info_payload, total_reward)

            self._episode_rewards.append(total_reward)
            self._episode_lengths.append(episode_length)
            self._episode_successes.append(1 if success else 0)

            self._metrics.episode_reward_mean = sum(self._episode_rewards) / len(self._episode_rewards)
            self._metrics.mean_reward = self._metrics.episode_reward_mean
            self._metrics.episode_length_mean = sum(self._episode_lengths) / len(self._episode_lengths)
            self._metrics.success_rate = sum(self._episode_successes) / len(self._episode_successes)

            if self._record_current_episode_trace:
                self.episode_captured.emit(self._build_episode_trace(info_payload, total_reward, success))

            self._episode_steps_buffer = []
            self._episode_moments_buffer = []
            self._episode_reward_total = 0.0
            self._episode_step_counter = 0
            self._reset_environment()

        if self._episode_successes:
            self._metrics.success_rate = sum(self._episode_successes) / len(self._episode_successes)

        if emit_metrics:
            self._emit_metrics_updated(now=now)

        if done:

            if self._config.max_episodes is not None and self._metrics.episode >= self._config.max_episodes:
                self._finish_run(TrainingStatus.FINISHED)
                return

            if self._evaluate_breakpoints():
                return
        elif self._evaluate_breakpoints():
            return

        if (
            self._config.max_duration_seconds is not None
            and (now - self._started_at) >= self._config.max_duration_seconds
        ):
            self._finish_run(TrainingStatus.FINISHED)
            return

        if self._config.max_steps is not None and self._metrics.step >= self._config.max_steps:
            self._finish_run(TrainingStatus.FINISHED)

    def _emit_metrics_updated(self, *, now: float | None = None) -> None:
        self._last_metrics_emitted_at = time.perf_counter() if now is None else now
        self.metrics_updated.emit(self._metrics)

    def _build_episode_trace(
        self,
        final_info: dict[str, Any],
        total_reward: float,
        success: bool,
    ) -> EpisodeTrace:
        steps = list(self._episode_steps_buffer)
        for idx, step in enumerate(steps):
            if step.terminated or step.truncated:
                steps = steps[: idx + 1]
                break

        moments = list(self._episode_moments_buffer)
        if len(moments) > len(steps) + 1:
            moments = moments[: len(steps) + 1]
        return EpisodeTrace(
            episode_id=self._metrics.episode,
            run_id=self._run_id,
            total_reward=total_reward,
            success=success,
            steps=steps,
            moments=moments,
            initial_observation=steps[0].observation if steps else None,
            task_snapshot=TaskSnapshot(
                environment_id=self._task.environment_id if self._task is not None else "unknown",
                task_name=self._task.name if self._task is not None else "unknown",
                task_id=self._task.task_id if self._task is not None else None,
                task_config=dict(self._task.config) if self._task is not None else {},
                reward_config=dict(self._task.reward_config) if self._task is not None else {},
                termination_config=(
                    dict(self._task.termination_config) if self._task is not None else {}
                ),
                metadata={
                    "seed": self._config.seed,
                    "episode_seed": self._episode_seed,
                    "run_id": self._run_id,
                },
            ),
            metadata={
                "runner": "gymnasium",
                "trace_sample_rate": self._config.episode_trace_sample_rate,
                "restorable_state_captured": any(
                    moment.restorable_env_state is not None for moment in moments
                ),
            },
        )

    def _episode_success(self, final_info: dict[str, Any], total_reward: float) -> bool:
        if final_info.get("forced_failure"):
            return False
        if "is_success" in final_info:
            return bool(final_info["is_success"])
        return total_reward > 0.0

    def _connect_environment_or_raise(self) -> None:
        if self._task is None:
            msg = "Cannot start training without a task."
            raise RuntimeError(msg)
        if self._env_factory is None:
            msg = f"Cannot start training for task '{self._task.name}': no environment factory was provided."
            raise RuntimeError(msg)

        try:
            candidate_env = self._env_factory(self._task)
        except Exception as exc:
            self._env = None
            msg = (
                f"Cannot create environment '{self._task.environment_id}' "
                f"for task '{self._task.name}'."
            )
            raise RuntimeError(msg) from exc

        if not self._is_env_compatible(candidate_env):
            try:
                close = getattr(candidate_env, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            self._env = None
            msg = (
                f"Environment '{self._task.environment_id}' for task '{self._task.name}' "
                "is not Gymnasium-compatible: expected reset(), step(), and action_space."
            )
            raise RuntimeError(msg)

        if (
            self._config.algorithm == "q_learning"
            and self._action_space_size(getattr(candidate_env, "action_space", None)) is None
        ):
            try:
                close = getattr(candidate_env, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            self._env = None
            msg = (
                f"Cannot start Q-learning for task '{self._task.name}': "
                "Q-learning requires a discrete action_space.n."
            )
            raise RuntimeError(msg)

        if is_stable_baselines3_algorithm(self._config.algorithm):
            self._connect_stable_baselines3_environment_or_raise(candidate_env)
            return

        self._env = candidate_env
        self._reset_environment()
        if self._observation is None:
            self._cleanup_environment()
            msg = (
                f"Environment '{self._task.environment_id}' for task '{self._task.name}' "
                "could not be reset."
            )
            raise RuntimeError(msg)

    def _connect_stable_baselines3_environment_or_raise(self, env: Any) -> None:
        assert self._task is not None
        trace_env = StableBaselines3TraceWrapper(
            env,
            task=self._task,
            config=self._config,
            run_id=self._run_id,
            trace_random=self._trace_random,
        )
        try:
            model = create_stable_baselines3_model(
                algorithm=self._config.algorithm,
                env=trace_env,
                config=self._config,
                learner_state=self._pending_sb3_learner_state,
            )
        except Exception as exc:
            try:
                trace_env.close()
            except Exception:
                pass
            self._env = None
            self._sb3_trace_env = None
            self._sb3_model = None
            msg = (
                f"Cannot initialize Stable-Baselines3 algorithm '{self._config.algorithm}' "
                f"for task '{self._task.name}'."
            )
            raise RuntimeError(msg) from exc

        self._env = trace_env
        self._sb3_trace_env = trace_env
        self._sb3_model = model

    def _is_env_compatible(self, env: Any) -> bool:
        return (
            env is not None
            and hasattr(env, "reset")
            and hasattr(env, "step")
            and hasattr(env, "action_space")
        )

    def _reset_environment(self) -> None:
        if self._env is None:
            self._observation = None
            self._record_current_episode_trace = False
            self._episode_steps_buffer = []
            self._episode_moments_buffer = []
            return

        try:
            reset_seed = None
            if self._config.seed is not None:
                reset_seed = self._config.seed + self._metrics.episode
            self._episode_seed = reset_seed
            if reset_seed is None:
                reset_result = self._env.reset()
            else:
                reset_result = self._env.reset(seed=reset_seed)
        except Exception:
            self._observation = None
            self._record_current_episode_trace = False
            self._episode_steps_buffer = []
            self._episode_moments_buffer = []
            return

        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, _info = reset_result
        else:
            observation = reset_result

        self._observation = self._normalize_observation(observation)
        self._episode_steps_buffer = []
        self._episode_moments_buffer = []
        self._record_current_episode_trace = self._should_record_episode_trace()
        if self._record_current_episode_trace:
            self._episode_moments_buffer.append(
                EpisodeMoment(
                    episode_id=self._current_episode_id(),
                    moment_index=0,
                    observation=self._observation,
                    restorable_env_state=self._maybe_export_restorable_state(self._env),
                    metadata={"phase": "initial"},
                )
            )

    def _normalize_observation(self, observation: Any) -> Any:
        if observation is None or isinstance(observation, (str, int, float, bool)):
            return observation
        if hasattr(observation, "item") and callable(getattr(observation, "item")):
            try:
                return observation.item()
            except Exception:
                pass
        return observation

    def _select_action(self, observation: Any, env: Any) -> int:
        action_space = getattr(env, "action_space", None)
        action_count = self._action_space_size(action_space)

        if action_count is not None and self._config.algorithm == "q_learning":
            if self._random.random() < self._metrics.exploration_rate:
                return self._random.randrange(action_count)

            state_key = self._state_key(observation)
            best_q = float("-inf")
            best_actions: list[int] = []
            for action in range(action_count):
                q_value = self._q_values.get((state_key, action), 0.0)
                if q_value > best_q:
                    best_q = q_value
                    best_actions = [action]
                elif q_value == best_q:
                    best_actions.append(action)
            return self._random.choice(best_actions) if best_actions else 0

        if action_space is not None and hasattr(action_space, "sample"):
            sampled = action_space.sample()
            try:
                return int(sampled)
            except (TypeError, ValueError):
                return 0

        return 0

    def _q_learning_exploration_rate(self) -> float:
        epsilon = max(0.0, min(1.0, float(self._config.epsilon)))
        epsilon_min = self._q_learning_hyperparameter_float(
            "epsilon_min",
            0.02 if epsilon > 0.0 else 0.0,
        )
        epsilon_min = max(0.0, min(epsilon, epsilon_min))
        completed_steps = max(0, self._metrics.step - 1)

        epsilon_decay = self._q_learning_hyperparameter_float("epsilon_decay", None)
        if epsilon_decay is not None:
            epsilon_decay = max(0.0, min(1.0, epsilon_decay))
            exploration_rate = epsilon * (epsilon_decay ** completed_steps)
            return max(epsilon_min, exploration_rate)

        max_episodes = self._config.max_episodes
        if max_episodes is not None:
            if max_episodes <= 1:
                return max(epsilon_min, epsilon)
            progress = min(
                1.0,
                max(0, self._metrics.episode) / max(max_episodes - 1, 1),
            )
            return max(epsilon_min, epsilon * (1.0 - progress))

        max_steps = self._config.max_steps
        if max_steps is None or max_steps <= 1:
            return max(epsilon_min, epsilon)

        progress = min(1.0, completed_steps / max(max_steps - 1, 1))
        return max(epsilon_min, epsilon * (1.0 - progress))

    def _q_learning_hyperparameter_float(self, key: str, default: float | None) -> float | None:
        raw_value = self._config.hyperparameters.get(key)
        if raw_value is None:
            return default
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return default

    def _model_updates_disabled(self) -> bool:
        metadata_flag = self._config.metadata.get("disable_model_updates")
        hyperparameter_flag = self._config.hyperparameters.get("disable_model_updates")
        return bool(metadata_flag or hyperparameter_flag)

    def _update_q_learning(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        env: Any,
    ) -> None:
        action_count = self._action_space_size(getattr(env, "action_space", None))
        if action_count is None:
            self._metrics.value_loss = None
            self._metrics.policy_loss = None
            return

        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        current_q = self._q_values.get((state_key, action), 0.0)
        if done:
            target = reward
        else:
            best_next = max(
                self._q_values.get((next_state_key, candidate_action), 0.0)
                for candidate_action in range(action_count)
            )
            target = reward + self._config.gamma * best_next

        td_error = target - current_q
        updated_q = current_q + self._config.learning_rate * td_error
        self._q_values[(state_key, action)] = updated_q

        self._metrics.value_loss = abs(td_error)
        self._metrics.policy_loss = None

    def _action_space_size(self, action_space: Any) -> int | None:
        if action_space is None or not hasattr(action_space, "n"):
            return None

        try:
            n_actions = int(action_space.n)
        except (TypeError, ValueError):
            return None

        if n_actions <= 0:
            return None
        return n_actions

    def _state_key(self, state: Any) -> str:
        if isinstance(state, (str, int, float, bool)) or state is None:
            return str(state)
        if hasattr(state, "tolist"):
            try:
                return repr(state.tolist())
            except Exception:
                return repr(state)
        return repr(state)

    def _finish_run(self, status: TrainingStatus) -> None:
        self._emit_metrics_updated()
        if status in {TrainingStatus.FINISHED, TrainingStatus.STOPPED}:
            self._stop_requested.set()
        if status in {TrainingStatus.FINISHED, TrainingStatus.STOPPED}:
            self._cleanup_environment()
        self._set_status(status)

    def _cleanup_environment(self) -> None:
        if self._env is None:
            return
        try:
            self._env.close()
        except Exception:
            pass
        self._env = None
        self._observation = None

    def _current_episode_id(self) -> int:
        return self._metrics.episode + 1

    def _should_record_episode_trace(self) -> bool:
        rate = self._config.episode_trace_sample_rate
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        return self._trace_random.random() < rate

    def _maybe_export_restorable_state(self, env: Any) -> Any | None:
        if env is None or not hasattr(env, "export_state"):
            return None
        export_state = getattr(env, "export_state")
        if not callable(export_state):
            return None
        try:
            return export_state()
        except Exception:
            return None

    def export_learner_state(self) -> dict[str, Any]:
        if is_stable_baselines3_algorithm(self._config.algorithm):
            return export_stable_baselines3_learner_state(
                self._sb3_model,
                algorithm=self._config.algorithm,
            )
        if self._config.algorithm != "q_learning":
            return {}
        return {
            "algorithm": "q_learning",
            "q_values": [
                {
                    "state_key": state_key,
                    "action": action,
                    "value": value,
                }
                for (state_key, action), value in sorted(self._q_values.items())
            ],
        }

    def _restore_checkpoint_state(self, checkpoint: Checkpoint | None) -> None:
        if checkpoint is None:
            return
        metadata = checkpoint.metadata if isinstance(checkpoint.metadata, dict) else {}
        learner_state = metadata.get("learner_state")
        if not isinstance(learner_state, dict):
            return
        if learner_state.get("algorithm") != self._config.algorithm:
            return
        if is_stable_baselines3_algorithm(self._config.algorithm):
            self._pending_sb3_learner_state = learner_state
            return
        if self._config.algorithm != "q_learning":
            return

        q_values_payload = learner_state.get("q_values", [])
        restored: dict[tuple[str, int], float] = {}
        if not isinstance(q_values_payload, list):
            return

        for entry in q_values_payload:
            if not isinstance(entry, dict):
                continue
            state_key = str(entry.get("state_key", ""))
            try:
                action = int(entry.get("action", 0))
                value = float(entry.get("value", 0.0))
            except (TypeError, ValueError):
                continue
            restored[(state_key, action)] = value

        self._q_values = restored

    def _evaluate_breakpoints(self, *, defer_stop: bool = False) -> bool:
        if not self._config.breakpoints:
            return False

        for idx, rule in enumerate(self._config.breakpoints):
            if idx in self._triggered_breakpoints:
                continue
            if self._rule_matches(rule):
                self._triggered_breakpoints.add(idx)
                self._emit_metrics_updated()
                event = BreakpointEvent(
                    breakpoint=rule,
                    step=self._metrics.step,
                    episode=self._metrics.episode,
                    message=self._build_breakpoint_message(rule),
                )
                self.breakpoint_triggered.emit(event)
                actions = set(rule.actions)
                if "stop" in actions:
                    if defer_stop:
                        self._sb3_pending_finish_status = TrainingStatus.STOPPED
                    else:
                        self._finish_run(TrainingStatus.STOPPED)
                    return True
                if "pause" in actions:
                    self._set_status(TrainingStatus.PAUSED)
                    return True

        return False

    def _rule_matches(self, rule: Breakpoint) -> bool:
        kind = rule.kind
        threshold = rule.value

        if kind == "max_step":
            return self._metrics.step >= threshold
        if kind == "episode_count_gte":
            return self._metrics.episode >= threshold
        if kind == "mean_reward_gte":
            metric = self._windowed_reward_mean(rule.window) if rule.window else self._metrics.mean_reward
            return metric >= threshold
        if kind == "episode_reward_mean_gte":
            return self._metrics.episode_reward_mean >= threshold
        if kind == "success_rate_gte":
            return self._metrics.success_rate >= threshold
        if kind == "exploration_lte":
            return self._metrics.exploration_rate <= threshold
        if kind == "value_loss_lte":
            return self._metrics.value_loss is not None and self._metrics.value_loss <= threshold
        if kind == "policy_loss_lte":
            return self._metrics.policy_loss is not None and self._metrics.policy_loss <= threshold
        return False

    def _windowed_reward_mean(self, window: int) -> float:
        if window <= 0 or not self._episode_rewards:
            return self._metrics.mean_reward
        n = min(window, len(self._episode_rewards))
        values = list(self._episode_rewards)[-n:]
        return sum(values) / len(values)

    def _build_breakpoint_message(self, rule: Breakpoint) -> str:
        if rule.kind == "max_step":
            return f"Breakpoint hit: max_step >= {rule.value:.0f}"
        if rule.kind == "episode_count_gte":
            return f"Breakpoint hit: episode_count >= {rule.value:.0f}"
        if rule.kind == "mean_reward_gte":
            return f"Breakpoint hit: mean_reward >= {rule.value:.3f}"
        if rule.kind == "episode_reward_mean_gte":
            return f"Breakpoint hit: episode_reward_mean >= {rule.value:.3f}"
        if rule.kind == "success_rate_gte":
            return f"Breakpoint hit: success_rate >= {rule.value * 100.0:.1f}%"
        if rule.kind == "exploration_lte":
            return f"Breakpoint hit: exploration_rate <= {rule.value * 100.0:.1f}%"
        if rule.kind == "value_loss_lte":
            return f"Breakpoint hit: value_loss <= {rule.value:.3f}"
        if rule.kind == "policy_loss_lte":
            return f"Breakpoint hit: policy_loss <= {rule.value:.3f}"
        return f"Breakpoint hit: {rule.kind}"
