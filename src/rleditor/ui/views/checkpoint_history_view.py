from __future__ import annotations

from dataclasses import dataclass
from html import escape
from math import hypot

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from rleditor.application.services import TrainingHistorySnapshot
from rleditor.core.models import Checkpoint, EpisodeTrace, TaskSnapshot, TrainingRun


@dataclass(slots=True)
class _LineageNode:
    node_id: str
    kind: str
    label: str
    sublabel: str
    checkpoint: Checkpoint | None
    rect: QRectF
    center: QPointF
    environment_id: str | None = None
    algorithm: str | None = None


@dataclass(slots=True)
class _LineageEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    target_checkpoint: Checkpoint
    source_checkpoint: Checkpoint | None
    run: TrainingRun | None
    task_snapshot: TaskSnapshot | None
    episodes: list[EpisodeTrace]
    source_point: QPointF
    target_point: QPointF


class CheckpointGraphWidget(QWidget):
    node_selected = Signal(object)
    edge_selected = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._snapshot = TrainingHistorySnapshot([], [], {}, {})
        self._nodes: dict[str, _LineageNode] = {}
        self._edges: list[_LineageEdge] = []
        self._selected_node_id: str | None = None
        self._selected_edge_id: str | None = None
        self._content_size = QSize(900, 520)
        self.setMinimumSize(220, 140)

    @property
    def selected_node_id(self) -> str | None:
        return self._selected_node_id

    @property
    def selected_edge_id(self) -> str | None:
        return self._selected_edge_id

    def node_for_id(self, node_id: str) -> _LineageNode | None:
        return self._nodes.get(node_id)

    def edge_for_id(self, edge_id: str) -> _LineageEdge | None:
        for edge in self._edges:
            if edge.edge_id == edge_id:
                return edge
        return None

    def select_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            self._selected_node_id = None
            return
        self._selected_node_id = node_id
        self._selected_edge_id = None
        self.update()

    def select_edge(self, edge_id: str) -> None:
        if self.edge_for_id(edge_id) is None:
            self._selected_edge_id = None
            return
        self._selected_edge_id = edge_id
        self.update()

    def clear_selection(self) -> None:
        self._selected_node_id = None
        self._selected_edge_id = None
        self.update()

    def set_history(self, snapshot: TrainingHistorySnapshot) -> None:
        self._snapshot = snapshot
        self._rebuild_layout()
        self.update()

    def sizeHint(self) -> QSize:
        return self._content_size

    def mousePressEvent(self, event) -> None:
        position = event.position()
        for node in reversed(list(self._nodes.values())):
            if node.rect.contains(position):
                self.select_node(node.node_id)
                self.node_selected.emit(node)
                return

        edge = self._edge_at(position)
        if edge is not None:
            self.select_edge(edge.edge_id)
            self.edge_selected.emit(edge)
            return

        self._selected_edge_id = None
        self.update()
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#fbfdff"))

        if not self._nodes:
            painter.setPen(QColor("#64748b"))
            painter.drawText(
                self.rect().adjusted(24, 24, -24, -24),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                "No checkpoints yet.\nStart training to create the first lineage.",
            )
            return

        for edge in self._edges:
            selected = self._selected_edge_id == edge.edge_id
            painter.setPen(QPen(QColor("#0f766e") if selected else QColor("#94a3b8"), 3 if selected else 2))
            painter.drawLine(edge.source_point, edge.target_point)

            label_rect = QRectF(
                min(edge.source_point.x(), edge.target_point.x()) - 90,
                (edge.source_point.y() + edge.target_point.y()) / 2.0 - 18,
                180,
                34,
            )
            painter.setPen(QColor("#0f172a") if selected else QColor("#475569"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, self._edge_label(edge))

        for node in self._nodes.values():
            selected = self._selected_node_id == node.node_id
            if node.kind == "root":
                fill = QColor("#f1f5f9")
                border = QColor("#0f766e") if selected else QColor("#94a3b8")
            else:
                fill = QColor("#dbeafe") if selected else QColor("#ffffff")
                border = QColor("#2563eb") if selected else QColor("#94a3b8")

            painter.setPen(QPen(border, 2))
            painter.setBrush(fill)
            painter.drawRoundedRect(node.rect, 12, 12)

            painter.setPen(QColor("#0f172a"))
            title_rect = node.rect.adjusted(10, 8, -10, -24)
            sub_rect = node.rect.adjusted(10, 28, -10, -8)
            painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, node.label)
            painter.setPen(QColor("#475569"))
            painter.drawText(sub_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, node.sublabel)

    def _rebuild_layout(self) -> None:
        self._nodes = {}
        self._edges = []

        checkpoints_by_id = {checkpoint.checkpoint_id: checkpoint for checkpoint in self._snapshot.checkpoints}
        runs_by_id = {run.run_id: run for run in self._snapshot.runs}

        root_ids: set[str] = set()
        children: dict[str, list[str]] = {}

        for checkpoint in self._snapshot.checkpoints:
            parent_id = checkpoint.parent_checkpoint_id
            if parent_id and parent_id in checkpoints_by_id:
                children.setdefault(parent_id, []).append(checkpoint.checkpoint_id)
                continue

            root_id = self._root_id_for_checkpoint(checkpoint)
            root_ids.add(root_id)
            children.setdefault(root_id, []).append(checkpoint.checkpoint_id)

        for parent_id, child_ids in children.items():
            child_ids.sort(key=lambda checkpoint_id: self._checkpoint_sort_key(checkpoints_by_id[checkpoint_id]))
            children[parent_id] = child_ids

        next_x = 0.0
        positions: dict[str, tuple[float, int]] = {}

        def assign(node_id: str, depth: int) -> float:
            nonlocal next_x
            child_ids = children.get(node_id, [])
            if not child_ids:
                x = next_x
                positions[node_id] = (x, depth)
                next_x += 1.0
                return x

            child_positions = [assign(child_id, depth + 1) for child_id in child_ids]
            x = sum(child_positions) / len(child_positions)
            positions[node_id] = (x, depth)
            return x

        ordered_root_ids = sorted(root_ids)
        for index, root_id in enumerate(ordered_root_ids):
            assign(root_id, 0)
            if index < len(ordered_root_ids) - 1:
                next_x += 1.0

        horizontal_gap = 260.0
        vertical_gap = 140.0
        left_margin = 120.0
        top_margin = 70.0

        for root_id in ordered_root_ids:
            slot_x, depth = positions[root_id]
            center = QPointF(left_margin + slot_x * horizontal_gap, top_margin + depth * vertical_gap)
            environment_id, algorithm = self._root_parts(root_id)
            rect = QRectF(center.x() - 86.0, center.y() - 28.0, 172.0, 56.0)
            self._nodes[root_id] = _LineageNode(
                node_id=root_id,
                kind="root",
                label="Untrained Agent",
                sublabel=f"{environment_id} | {algorithm}",
                checkpoint=None,
                rect=rect,
                center=center,
                environment_id=environment_id,
                algorithm=algorithm,
            )

        for checkpoint in self._snapshot.checkpoints:
            slot_x, depth = positions.get(checkpoint.checkpoint_id, (0.0, 1))
            center = QPointF(left_margin + slot_x * horizontal_gap, top_margin + depth * vertical_gap)
            rect = QRectF(center.x() - 92.0, center.y() - 34.0, 184.0, 68.0)
            self._nodes[checkpoint.checkpoint_id] = _LineageNode(
                node_id=checkpoint.checkpoint_id,
                kind="checkpoint",
                label=self._checkpoint_title(checkpoint),
                sublabel=f"ep {checkpoint.episode} | step {checkpoint.step}",
                checkpoint=checkpoint,
                rect=rect,
                center=center,
            )

        for checkpoint in sorted(self._snapshot.checkpoints, key=self._checkpoint_sort_key):
            source_node_id = checkpoint.parent_checkpoint_id
            source_checkpoint = None
            if source_node_id is not None and source_node_id in checkpoints_by_id:
                source_checkpoint = checkpoints_by_id[source_node_id]
            else:
                source_node_id = self._root_id_for_checkpoint(checkpoint)

            source_node = self._nodes.get(source_node_id)
            target_node = self._nodes.get(checkpoint.checkpoint_id)
            if source_node is None or target_node is None:
                continue

            run = runs_by_id.get(checkpoint.run_id or "")
            task_snapshot = self._snapshot.run_task_snapshots.get(checkpoint.run_id or "") or checkpoint.task_snapshot
            self._edges.append(
                _LineageEdge(
                    edge_id=f"edge:{checkpoint.checkpoint_id}",
                    source_node_id=source_node_id,
                    target_node_id=checkpoint.checkpoint_id,
                    target_checkpoint=checkpoint,
                    source_checkpoint=source_checkpoint,
                    run=run,
                    task_snapshot=task_snapshot,
                    episodes=self._episodes_for_edge(checkpoint, source_checkpoint),
                    source_point=QPointF(source_node.center.x(), source_node.rect.bottom()),
                    target_point=QPointF(target_node.center.x(), target_node.rect.top()),
                )
            )

        width = int(left_margin * 2 + max(1.0, next_x) * horizontal_gap)
        max_depth = max((depth for _, depth in positions.values()), default=0)
        height = int(top_margin * 2 + (max_depth + 1) * vertical_gap)
        self._content_size = QSize(max(320, width), max(240, height))
        self.resize(self._content_size)
        self.updateGeometry()

    def _edge_at(self, point: QPointF) -> _LineageEdge | None:
        for edge in reversed(self._edges):
            if self._distance_to_segment(point, edge.source_point, edge.target_point) <= 10.0:
                return edge
        return None

    def _distance_to_segment(self, point: QPointF, start: QPointF, end: QPointF) -> float:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        if dx == 0.0 and dy == 0.0:
            return hypot(point.x() - start.x(), point.y() - start.y())
        projection = (
            ((point.x() - start.x()) * dx + (point.y() - start.y()) * dy)
            / (dx * dx + dy * dy)
        )
        projection = max(0.0, min(1.0, projection))
        nearest_x = start.x() + projection * dx
        nearest_y = start.y() + projection * dy
        return hypot(point.x() - nearest_x, point.y() - nearest_y)

    def _checkpoint_title(self, checkpoint: Checkpoint) -> str:
        checkpoint_id = checkpoint.checkpoint_id.replace("_", " ").title()
        if len(checkpoint_id) > 24:
            return checkpoint_id[:21] + "..."
        return checkpoint_id

    def _checkpoint_sort_key(self, checkpoint: Checkpoint) -> tuple[str, str, int, int, str]:
        return (
            checkpoint.created_at,
            checkpoint.run_id or "",
            checkpoint.episode,
            checkpoint.step,
            checkpoint.checkpoint_id,
        )

    def _root_id_for_checkpoint(self, checkpoint: Checkpoint) -> str:
        task_snapshot = checkpoint.task_snapshot
        environment_id = task_snapshot.environment_id if task_snapshot is not None else "unknown"
        algorithm = str(checkpoint.metadata.get("algorithm", "unknown"))
        return f"root:{environment_id}:{algorithm}"

    def _root_parts(self, root_id: str) -> tuple[str, str]:
        _prefix, environment_id, algorithm = root_id.split(":", 2)
        return environment_id, algorithm

    def _episodes_for_edge(
        self,
        target_checkpoint: Checkpoint,
        source_checkpoint: Checkpoint | None,
    ) -> list[EpisodeTrace]:
        if target_checkpoint.run_id is None:
            return []
        traces = self._snapshot.episodes_by_run.get(target_checkpoint.run_id, [])
        start_episode = 0
        if source_checkpoint is not None and source_checkpoint.run_id == target_checkpoint.run_id:
            start_episode = source_checkpoint.episode
        return [
            trace
            for trace in traces
            if start_episode < trace.episode_id <= target_checkpoint.episode
        ]

    def _edge_label(self, edge: _LineageEdge) -> str:
        task_name = edge.task_snapshot.task_name if edge.task_snapshot is not None else "Training Run"
        if len(task_name) > 26:
            task_name = task_name[:23] + "..."
        return task_name


class CheckpointHistoryView(QWidget):
    inspect_episode_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._snapshot = TrainingHistorySnapshot([], [], {}, {})
        self._current_segment_episodes: list[EpisodeTrace] = []
        self._selected_start_node_id: str | None = None

        root = QVBoxLayout(self)

        title = QLabel("Checkpoint History")
        title.setObjectName("TitleLabel")
        subtitle = QLabel(
            "Browse checkpoint lineage top-down, inspect each training run, and jump directly to recorded episodes."
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        left_panel = QWidget(splitter)
        left_layout = QVBoxLayout(left_panel)
        left_group = QGroupBox("Checkpoint Lineage", left_panel)
        left_group_layout = QVBoxLayout(left_group)
        self.graph_widget = CheckpointGraphWidget()
        graph_scroll = QScrollArea(left_group)
        graph_scroll.setWidgetResizable(False)
        graph_scroll.setWidget(self.graph_widget)
        left_group_layout.addWidget(graph_scroll)
        left_layout.addWidget(left_group)

        right_panel = QWidget(splitter)
        right_layout = QVBoxLayout(right_panel)

        self.training_source_label = QLabel("Training start checkpoint: scratch")
        self.training_source_label.setWordWrap(True)
        self.selection_label = QLabel("Select a training edge to inspect a run.")
        self.selection_label.setWordWrap(True)

        checkpoint_group = QGroupBox("Node Details", right_panel)
        checkpoint_layout = QFormLayout(checkpoint_group)
        self.checkpoint_details = QTextEdit(checkpoint_group)
        self.checkpoint_details.setReadOnly(True)
        self.checkpoint_details.setMinimumHeight(80)
        checkpoint_layout.addRow(self.checkpoint_details)

        segment_group = QGroupBox("Training Run", right_panel)
        segment_layout = QVBoxLayout(segment_group)
        self.segment_details = QTextEdit(segment_group)
        self.segment_details.setReadOnly(True)
        self.segment_details.setMinimumHeight(80)
        self.episode_list = QListWidget(segment_group)
        self.episode_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.inspect_episode_button = QPushButton("Inspect Selected Episode", segment_group)
        self.inspect_episode_button.setEnabled(False)

        segment_layout.addWidget(self.segment_details)
        segment_layout.addWidget(self.episode_list, 1)
        segment_layout.addWidget(self.inspect_episode_button)

        right_layout.addWidget(self.training_source_label)
        right_layout.addWidget(self.selection_label)
        right_layout.addWidget(checkpoint_group)
        right_layout.addWidget(segment_group, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([620, 420])

        root.addWidget(title)
        root.addWidget(subtitle)
        root.addWidget(splitter, 1)

        self.graph_widget.node_selected.connect(self._show_node_details)
        self.graph_widget.edge_selected.connect(self._show_edge_details)
        self.episode_list.currentRowChanged.connect(self._on_episode_selection_changed)
        self.inspect_episode_button.clicked.connect(self._emit_inspect_selected_episode)

        self._render_empty_selection()

    def selected_checkpoint(self) -> Checkpoint | None:
        node_id = self._selected_start_node_id
        if node_id is None:
            return None
        node = self.graph_widget.node_for_id(node_id)
        if node is None:
            return None
        return node.checkpoint

    def start_from_scratch_selected(self) -> bool:
        node_id = self._selected_start_node_id
        if node_id is None:
            return not self._snapshot.checkpoints
        node = self.graph_widget.node_for_id(node_id)
        return node is not None and node.kind == "root"

    def set_history(self, snapshot: TrainingHistorySnapshot) -> None:
        selected_node_id = self._selected_start_node_id
        selected_edge_id = self.graph_widget.selected_edge_id
        self._snapshot = snapshot
        self.graph_widget.set_history(snapshot)

        resolved_node = None
        if selected_node_id is not None:
            resolved_node = self.graph_widget.node_for_id(selected_node_id)
        if resolved_node is None:
            latest_checkpoint = self._latest_checkpoint()
            if latest_checkpoint is not None:
                selected_node_id = latest_checkpoint.checkpoint_id
                resolved_node = self.graph_widget.node_for_id(selected_node_id)
        self._selected_start_node_id = selected_node_id if resolved_node is not None else None

        if resolved_node is not None:
            self.graph_widget.select_node(resolved_node.node_id)
            self._show_node_details(resolved_node)
        else:
            self._render_empty_selection()

        if selected_edge_id is not None:
            edge = self.graph_widget.edge_for_id(selected_edge_id)
            if edge is not None:
                self.graph_widget.select_edge(selected_edge_id)
                self._show_edge_details(edge)
                return

        self._render_empty_segment_selection()

    def _render_empty_selection(self) -> None:
        if self._snapshot.checkpoints:
            self.training_source_label.setText("Training start checkpoint: pending selection")
        else:
            self.training_source_label.setText("Training start checkpoint: scratch")
        self.selection_label.setText(
            f"History contains {len(self._snapshot.checkpoints)} checkpoint(s) across {len(self._snapshot.runs)} run(s). "
            "Select a training edge to inspect a run."
        )
        self.checkpoint_details.setHtml(
            self._details_panel_html(
                heading="Node Details",
                rows=[],
                empty_message="Checkpoint or root-node details will appear here.",
            )
        )
        self.segment_details.setPlainText("Training run details and recorded episodes will appear here.")
        self.episode_list.clear()
        self._current_segment_episodes = []
        self.inspect_episode_button.setEnabled(False)

    def _render_empty_segment_selection(self) -> None:
        self.selection_label.setText("Select a training edge to inspect a run.")
        self.segment_details.setPlainText("Training run details and recorded episodes will appear here.")
        self.episode_list.clear()
        self._current_segment_episodes = []
        self.inspect_episode_button.setEnabled(False)

    def _show_node_details(self, node: _LineageNode) -> None:
        self._selected_start_node_id = node.node_id
        if node.kind == "root":
            self.training_source_label.setText("Training start checkpoint: scratch")
            self._set_root_details(node)
            self._render_empty_segment_selection()
            return

        checkpoint = node.checkpoint
        if checkpoint is None:
            self._render_empty_selection()
            return

        self.training_source_label.setText(f"Training start checkpoint: {checkpoint.label}")
        self._set_checkpoint_details(checkpoint, heading="Selected checkpoint")
        self.segment_details.setPlainText("Select a training edge to inspect the run that produced this checkpoint.")
        self.episode_list.clear()
        self._current_segment_episodes = []
        self.inspect_episode_button.setEnabled(False)

    def _show_edge_details(self, edge: _LineageEdge) -> None:
        checkpoint = edge.target_checkpoint
        task_snapshot = edge.task_snapshot or checkpoint.task_snapshot
        task_name = task_snapshot.task_name if task_snapshot is not None else (checkpoint.task_name or "unknown")
        environment_id = task_snapshot.environment_id if task_snapshot is not None else "unknown"
        run = edge.run
        self.selection_label.setText(
            f"Selected training run: {run.run_id if run is not None else checkpoint.run_id or 'unknown'}"
        )
        self._set_checkpoint_details(edge.target_checkpoint, heading="Checkpoint produced by this run")
        start_label = edge.source_checkpoint.checkpoint_id if edge.source_checkpoint is not None else "scratch"

        lines = [
            f"Run ID: {run.run_id if run is not None else checkpoint.run_id or 'unknown'}",
            f"Start node: {start_label}",
            f"Target checkpoint: {checkpoint.checkpoint_id}",
            f"Task: {task_name}",
            f"Environment: {environment_id}",
            f"Recorded episodes available here: {len(edge.episodes)}",
        ]
        if run is not None:
            lines.extend(
                [
                    f"Run status: {run.status.value}",
                    f"Started at: {run.started_at or 'unknown'}",
                    f"Ended at: {run.ended_at or 'in progress'}",
                    f"Parent checkpoint: {run.parent_checkpoint_id or 'scratch'}",
                    f"Algorithm: {run.metadata.get('algorithm', 'unknown')}",
                    f"Seed: {run.metadata.get('seed', 'unknown')}",
                ]
            )
        self.segment_details.setPlainText("\n".join(lines))

        self.episode_list.clear()
        self._current_segment_episodes = list(edge.episodes)
        for trace in self._current_segment_episodes:
            self.episode_list.addItem(
                QListWidgetItem(
                    f"Episode {trace.episode_id} | reward={trace.total_reward:.2f} | success={trace.success}"
                )
            )

        if self._current_segment_episodes:
            self.episode_list.setCurrentRow(0)
            self.inspect_episode_button.setEnabled(True)
        else:
            self.inspect_episode_button.setEnabled(False)

    def _on_episode_selection_changed(self, row: int) -> None:
        self.inspect_episode_button.setEnabled(0 <= row < len(self._current_segment_episodes))

    def _emit_inspect_selected_episode(self) -> None:
        row = self.episode_list.currentRow()
        if row < 0 or row >= len(self._current_segment_episodes):
            return
        self.inspect_episode_requested.emit(self._current_segment_episodes[row])

    def _latest_checkpoint(self) -> Checkpoint | None:
        if not self._snapshot.checkpoints:
            return None
        return self._snapshot.checkpoints[-1]

    def _set_root_details(self, node: _LineageNode) -> None:
        self.checkpoint_details.setHtml(
            self._details_panel_html(
                heading="Selected training origin",
                rows=[
                    ("Node type", "Untrained agent root"),
                    ("Environment", node.environment_id or "unknown"),
                    ("Algorithm", node.algorithm or "unknown"),
                ],
                note="Training will start from scratch with no checkpoint restoration.",
            )
        )

    def _set_checkpoint_details(self, checkpoint: Checkpoint, *, heading: str) -> None:
        task_snapshot = checkpoint.task_snapshot
        environment_id = task_snapshot.environment_id if task_snapshot is not None else "unknown"
        task_name = task_snapshot.task_name if task_snapshot is not None else (checkpoint.task_name or "unknown")
        rows = [
            ("Checkpoint ID", checkpoint.checkpoint_id),
            ("Created at", checkpoint.created_at),
            ("Reason", checkpoint.reason),
            ("Run ID", checkpoint.run_id or "unknown"),
            ("Task", task_name),
            ("Environment", environment_id),
            ("Episode", str(checkpoint.episode)),
            ("Step", str(checkpoint.step)),
            ("Parent checkpoint", checkpoint.parent_checkpoint_id or "scratch"),
        ]
        metrics = checkpoint.metadata.get("training_metrics")
        metric_rows: list[tuple[str, str]] = []
        if isinstance(metrics, dict) and metrics:
            metric_rows = [
                ("Mean reward", f"{float(metrics.get('mean_reward', 0.0)):.3f}"),
                ("Success rate", f"{float(metrics.get('success_rate', 0.0)) * 100.0:.1f}%"),
                ("Episode return (mean)", f"{float(metrics.get('episode_reward_mean', 0.0)):.3f}"),
                ("Episode length (mean)", f"{float(metrics.get('episode_length_mean', 0.0)):.3f}"),
                ("Exploration rate", f"{float(metrics.get('exploration_rate', 0.0)) * 100.0:.1f}%"),
                ("TD error", self._format_metric_value(metrics.get("value_loss"))),
                ("Policy loss", self._format_metric_value(metrics.get("policy_loss"))),
            ]
        else:
            metric_rows = []

        self.checkpoint_details.setHtml(
            self._details_panel_html(
                heading=heading,
                rows=rows,
                metrics_heading="Recorded training metrics",
                metric_rows=metric_rows,
                metrics_empty_message="No performance snapshot is available for this checkpoint.",
            )
        )

    def _details_panel_html(
        self,
        *,
        heading: str,
        rows: list[tuple[str, str]],
        note: str | None = None,
        metrics_heading: str | None = None,
        metric_rows: list[tuple[str, str]] | None = None,
        metrics_empty_message: str | None = None,
        empty_message: str | None = None,
    ) -> str:
        parts = [
            "<div style='font-family: Sans-Serif; color: #0f172a;'>",
            f"<div style='font-weight: 700; margin-bottom: 8px;'>{escape(heading)}</div>",
        ]

        if rows:
            parts.append(self._table_html(rows))
        elif empty_message is not None:
            parts.append(f"<div style='color: #64748b;'>{escape(empty_message)}</div>")

        if note is not None:
            parts.append(
                f"<div style='margin-top: 10px; color: #475569;'>{escape(note)}</div>"
            )

        if metrics_heading is not None:
            parts.append(
                f"<div style='font-weight: 700; margin: 12px 0 8px 0;'>{escape(metrics_heading)}</div>"
            )
            if metric_rows:
                parts.append(self._table_html(metric_rows))
            elif metrics_empty_message is not None:
                parts.append(f"<div style='color: #64748b;'>{escape(metrics_empty_message)}</div>")

        parts.append("</div>")
        return "".join(parts)

    def _table_html(self, rows: list[tuple[str, str]]) -> str:
        table_rows = []
        for label, value in rows:
            table_rows.append(
                "<tr>"
                f"<td style='padding: 6px 10px; border: 1px solid #dbe4f0; background: #f8fafc; "
                f"font-weight: 600; color: #334155; width: 42%;'>{escape(label)}</td>"
                f"<td style='padding: 6px 10px; border: 1px solid #dbe4f0; color: #0f172a;'>{escape(value)}</td>"
                "</tr>"
            )
        return (
            "<table cellspacing='0' cellpadding='0' "
            "style='width: 100%; border-collapse: collapse; background: #ffffff;'>"
            f"{''.join(table_rows)}"
            "</table>"
        )

    def _format_metric_value(self, value: object) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return str(value)
