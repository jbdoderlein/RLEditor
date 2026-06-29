# SpaceTimeRL

SpaceTimeRL is a PySide6 application for building, training, evaluating, and inspecting reinforcement-learning tasks and checkpoint histories. The current built-in environments focus on Frozen Lake and Blackjack, with Q-learning and Stable-Baselines3 backends.

## Replication information

The `eval` directory contains all curricula and tasks used to generate the figures and tables in the paper. To replicate the experiments, you can start tha app and use the import button to load baseline curricula (`eval/generated/baselines`) or author-created curricula (`eval/curricula_author1` and `eval/curricula_author2`).

`eval/trainin_report_cumulative_reward.ipynb` is a Jupyter notebook that produces every figure and table in the paper, using the results from curricula and tasks in `eval/generated/baselines` and `eval/curricula_author1`/`eval/curricula_author2`.

All experiments are run with fixed seed `121` to ensure reproducibility.

## Run

Run the app:

```bash
uv run rleditor --env frozen_lake
```

Useful options:

```bash
uv run rleditor --env frozen_lake --seed 121
```

Run tests:

```bash
uv run --group dev pytest -q
```

## Repository Structure

```text
.
├── src/rleditor/              Main application package
├── src/eval/                  Evaluation and curriculum-generation scripts/assets
├── tests/                     Pytest test suite
├── eval/                      Experiment outputs, figures, tables, exported reports
├── scripts/                   Small command wrappers/helpers
├── icon.png                   Application icon used by Qt/GNOME integration
├── pyproject.toml             Package metadata, dependencies, console scripts
└── uv.lock                    Locked dependency resolution
```

## Application Package

`src/rleditor/` is split by responsibility:

```text
src/rleditor/
├── app.py                     CLI argument handling and QApplication bootstrap
├── main.py                    Console-script entry point
├── core/models.py             Shared dataclasses: tasks, runs, checkpoints, traces, configs
├── application/               App services, persistence, global randomness
├── infra/                     Training/evaluation runners and ML backend adapters
├── plugins/                   Plugin interfaces and built-in environments
├── ui/                        PySide widgets, main window, views, theme, logging
└── tools/                     Auxiliary tools such as the interaction timeline viewer
```

Key subdirectories:

- `application/services.py`: orchestration for training runs, checkpoints, evaluation, imports, and history.
- `application/persistence.py`: project save/load and checkpoint state sidecar files.
- `application/randomness.py`: global app seed helpers used by training and generated maps.
- `infra/training_runner.py`: Q-learning training loop and episode capture.
- `infra/evaluation_runner.py`: policy evaluation and trace generation.
- `infra/stable_baselines_backend.py`: Stable-Baselines3 integration.
- `plugins/base.py` and `plugins/registry.py`: environment plugin contract and registry.
- `plugins/builtin/`: Frozen Lake and Blackjack plugins.
- `ui/shell/main_window.py`: top-level UI wiring.
- `ui/views/`: task editor/history, training monitor, evaluation, checkpoint history, and episode inspector views.

## Tests

Tests are organized by module or behavior:

- `tests/test_main_window.py`: high-level UI orchestration.
- `tests/test_checkpoint_history_view.py`: checkpoint graph/history interactions.
- `tests/test_training_runner_logic.py` and `tests/test_q_learning.py`: training-loop behavior.
- `tests/test_frozen_lake_env.py` and `tests/test_blackjack_env.py`: built-in environment behavior.
- `tests/test_persistence.py`: project storage.
- `tests/test_app.py`: CLI parser and startup helpers.

Most UI tests run with `QT_QPA_PLATFORM=offscreen`.
