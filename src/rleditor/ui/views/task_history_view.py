from __future__ import annotations

from dataclasses import dataclass
from pprint import pformat

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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

from rleditor.core.models import DerivedTaskDefinition, TaskDefinition


@dataclass(slots=True)
class _TaskNode:
    node_id: str
    workspace_index: int
    task: TaskDefinition
    rect: QRectF
    center: QPointF
    is_derived: bool
    parent_workspace_index: int | None = None


@dataclass(slots=True)
class _TaskEdge:
    source_node_id: str
    target_node_id: str
    source_point: QPointF
    target_point: QPointF


class TaskGraphWidget(QWidget):
    selection_changed = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
        self._tasks: list[TaskDefinition] = []
        self._nodes: dict[str, _TaskNode] = {}
        self._edges: list[_TaskEdge] = []
        self._selected_node_ids: set[str] = set()
        self._primary_node_id: str | None = None
        self._content_size = QSize(900, 480)
        self.setMinimumSize(220, 140)

    @property
    def primary_node_id(self) -> str | None:
        return self._primary_node_id

    def sizeHint(self) -> QSize:
        return self._content_size

    def node_for_id(self, node_id: str) -> _TaskNode | None:
        return self._nodes.get(node_id)

    def primary_workspace_index(self) -> int | None:
        if self._primary_node_id is None:
            return None
        node = self._nodes.get(self._primary_node_id)
        if node is None:
            return None
        return node.workspace_index

    def selected_workspace_indexes(self) -> list[int]:
        nodes = [self._nodes[node_id] for node_id in self._selected_node_ids if node_id in self._nodes]
        nodes.sort(key=lambda node: node.workspace_index)
        return [node.workspace_index for node in nodes]

    def set_tasks(self, tasks: list[TaskDefinition]) -> None:
        previous_selected = set(self._selected_node_ids)
        previous_primary = self._primary_node_id
        self._tasks = list(tasks)
        self._rebuild_layout()

        self._selected_node_ids = {
            node_id for node_id in previous_selected if node_id in self._nodes
        }
        if previous_primary is not None and previous_primary in self._nodes:
            self._primary_node_id = previous_primary
            self._selected_node_ids.add(previous_primary)
        elif self._selected_node_ids:
            self._primary_node_id = self._first_selected_node_id()
        elif self._nodes:
            self._primary_node_id = self._node_id_for_workspace_index(0)
            if self._primary_node_id is not None:
                self._selected_node_ids = {self._primary_node_id}
        else:
            self._primary_node_id = None

        self.update()

    def set_primary_workspace_index(
        self,
        index: int,
        *,
        preserve_multi_selection: bool,
        emit_signal: bool,
    ) -> None:
        node_id = self._node_id_for_workspace_index(index)
        if node_id is None:
            return
        if preserve_multi_selection:
            self._selected_node_ids.add(node_id)
        else:
            self._selected_node_ids = {node_id}
        self._primary_node_id = node_id
        self.update()
        if emit_signal:
            self._emit_selection_changed()

    def toggle_workspace_index_selection(self, index: int, *, emit_signal: bool) -> None:
        node_id = self._node_id_for_workspace_index(index)
        if node_id is None:
            return
        self._toggle_node_selection(node_id)
        self.update()
        if emit_signal:
            self._emit_selection_changed()

    def clear_selection(self, *, emit_signal: bool) -> None:
        self._selected_node_ids.clear()
        self._primary_node_id = None
        self.update()
        if emit_signal:
            self._emit_selection_changed()

    def mousePressEvent(self, event) -> None:
        position = event.position()
        for node in reversed(list(self._nodes.values())):
            if node.rect.contains(position):
                ctrl_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                if ctrl_pressed:
                    self._toggle_node_selection(node.node_id)
                else:
                    self._selected_node_ids = {node.node_id}
                    self._primary_node_id = node.node_id
                self.update()
                self._emit_selection_changed()
                return

        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.clear_selection(emit_signal=True)
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
                "No tasks yet.\nCreate or derive a task to build the task lineage.",
            )
            return

        for edge in self._edges:
            painter.setPen(QPen(QColor("#94a3b8"), 2))
            painter.drawLine(edge.source_point, edge.target_point)

        for node in self._nodes.values():
            selected = node.node_id in self._selected_node_ids
            primary = node.node_id == self._primary_node_id

            if node.is_derived:
                fill = QColor("#fef3c7") if selected else QColor("#fffdf5")
                border = QColor("#d97706") if primary else QColor("#cbd5e1")
            else:
                fill = QColor("#dbeafe") if selected else QColor("#ffffff")
                border = QColor("#2563eb") if primary else QColor("#94a3b8")

            painter.setPen(QPen(border, 3 if primary else 2))
            painter.setBrush(fill)
            painter.drawRoundedRect(node.rect, 12, 12)

            painter.setPen(QColor("#0f172a"))
            title_rect = node.rect.adjusted(10, 8, -10, -24)
            sub_rect = node.rect.adjusted(10, 28, -10, -8)
            painter.drawText(
                title_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._node_title(node),
            )
            painter.setPen(QColor("#475569"))
            painter.drawText(
                sub_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                self._node_subtitle(node),
            )

    def _emit_selection_changed(self) -> None:
        primary_node = self._nodes.get(self._primary_node_id) if self._primary_node_id is not None else None
        selected_nodes = [
            self._nodes[node_id]
            for node_id in self._selected_node_ids
            if node_id in self._nodes
        ]
        selected_nodes.sort(key=lambda node: node.workspace_index)
        self.selection_changed.emit(primary_node, selected_nodes)

    def _toggle_node_selection(self, node_id: str) -> None:
        if node_id in self._selected_node_ids:
            if len(self._selected_node_ids) == 1:
                self._primary_node_id = node_id
                return
            self._selected_node_ids.remove(node_id)
            if self._primary_node_id == node_id:
                self._primary_node_id = self._first_selected_node_id()
            return

        self._selected_node_ids.add(node_id)
        self._primary_node_id = node_id

    def _first_selected_node_id(self) -> str | None:
        for node in sorted(self._nodes.values(), key=lambda item: item.workspace_index):
            if node.node_id in self._selected_node_ids:
                return node.node_id
        return None

    def _node_id_for_workspace_index(self, index: int) -> str | None:
        node_id = f"task:{index}"
        if node_id in self._nodes:
            return node_id
        return None

    def _rebuild_layout(self) -> None:
        self._nodes = {}
        self._edges = []

        if not self._tasks:
            self._content_size = QSize(900, 480)
            self.resize(self._content_size)
            self.updateGeometry()
            return

        identifier_to_index: dict[str, int] = {}
        for index, task in enumerate(self._tasks):
            if task.task_id:
                identifier_to_index[str(task.task_id)] = index
            identifier_to_index.setdefault(task.name, index)

        children: dict[int | None, list[int]] = {}
        parent_by_index: dict[int, int | None] = {}
        roots: list[int] = []

        for index, task in enumerate(self._tasks):
            parent_index = self._parent_workspace_index(task, identifier_to_index)
            parent_by_index[index] = parent_index
            if parent_index is None:
                roots.append(index)
                continue
            children.setdefault(parent_index, []).append(index)

        for child_indexes in children.values():
            child_indexes.sort()

        next_x = 0.0
        positions: dict[int, tuple[float, int]] = {}

        def assign(task_index: int, depth: int) -> float:
            nonlocal next_x
            child_indexes = children.get(task_index, [])
            if not child_indexes:
                x = next_x
                positions[task_index] = (x, depth)
                next_x += 1.0
                return x

            child_positions = [assign(child_index, depth + 1) for child_index in child_indexes]
            x = sum(child_positions) / len(child_positions)
            positions[task_index] = (x, depth)
            return x

        for root_position, root_index in enumerate(roots):
            assign(root_index, 0)
            if root_position < len(roots) - 1:
                next_x += 1.0

        horizontal_gap = 260.0
        vertical_gap = 140.0
        left_margin = 120.0
        top_margin = 70.0

        for index, task in enumerate(self._tasks):
            slot_x, depth = positions.get(index, (0.0, 0))
            center = QPointF(left_margin + slot_x * horizontal_gap, top_margin + depth * vertical_gap)
            rect = QRectF(center.x() - 96.0, center.y() - 34.0, 192.0, 68.0)
            node_id = f"task:{index}"
            self._nodes[node_id] = _TaskNode(
                node_id=node_id,
                workspace_index=index,
                task=task,
                rect=rect,
                center=center,
                is_derived=isinstance(task, DerivedTaskDefinition),
                parent_workspace_index=parent_by_index.get(index),
            )

        for index, task in enumerate(self._tasks):
            parent_index = parent_by_index.get(index)
            if parent_index is None:
                continue
            source_node = self._nodes.get(f"task:{parent_index}")
            target_node = self._nodes.get(f"task:{index}")
            if source_node is None or target_node is None:
                continue
            self._edges.append(
                _TaskEdge(
                    source_node_id=source_node.node_id,
                    target_node_id=target_node.node_id,
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

    def _parent_workspace_index(
        self,
        task: TaskDefinition,
        identifier_to_index: dict[str, int],
    ) -> int | None:
        if not isinstance(task, DerivedTaskDefinition):
            return None
        parent_task_id = task.parent_task_id
        if not parent_task_id:
            return None
        return identifier_to_index.get(str(parent_task_id))

    def _node_title(self, node: _TaskNode) -> str:
        title = node.task.name
        if len(title) > 26:
            return title[:23] + "..."
        return title

    def _node_subtitle(self, node: _TaskNode) -> str:
        prefix = "Derived Task" if node.is_derived else "Task"
        environment_id = node.task.environment_id or "unknown"
        return f"{prefix} | {environment_id}"


class TaskHistoryView(QWidget):
    selection_changed = Signal(object, object)
    create_task_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._tasks: list[TaskDefinition] = []

        root = QVBoxLayout(self)

        title = QLabel("Task History")
        title.setObjectName("TitleLabel")
        add_task_button = QPushButton("Add New Task", self)
        add_task_button.clicked.connect(self.create_task_requested.emit)

        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(add_task_button)

        subtitle = QLabel(
            "Browse the task forest, choose the primary training task, and Ctrl-click to prepare multiple tasks."
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        left_panel = QWidget(splitter)
        left_layout = QVBoxLayout(left_panel)
        left_group = QGroupBox("Task Lineage", left_panel)
        left_group_layout = QVBoxLayout(left_group)
        self.graph_widget = TaskGraphWidget()
        graph_scroll = QScrollArea(left_group)
        graph_scroll.setWidgetResizable(False)
        graph_scroll.setWidget(self.graph_widget)
        left_group_layout.addWidget(graph_scroll)
        left_layout.addWidget(left_group)

        right_panel = QWidget(splitter)
        right_layout = QVBoxLayout(right_panel)

        self.training_task_label = QLabel("Training task: -")
        self.training_task_label.setWordWrap(True)
        self.selection_label = QLabel("Selected tasks: 0")
        self.selection_label.setWordWrap(True)

        details_group = QGroupBox("Task Details", right_panel)
        details_layout = QFormLayout(details_group)
        self.task_details = QTextEdit(details_group)
        self.task_details.setReadOnly(True)
        self.task_details.setMinimumHeight(100)
        details_layout.addRow(self.task_details)

        selection_group = QGroupBox("Multi-Selection", right_panel)
        selection_layout = QVBoxLayout(selection_group)
        self.selection_list = QListWidget(selection_group)
        selection_layout.addWidget(self.selection_list)

        right_layout.addWidget(self.training_task_label)
        right_layout.addWidget(self.selection_label)
        right_layout.addWidget(details_group)
        right_layout.addWidget(selection_group, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([620, 420])

        root.addLayout(title_row)
        root.addWidget(subtitle)
        root.addWidget(splitter, 1)

        self.graph_widget.selection_changed.connect(self._on_graph_selection_changed)
        self._render_empty_selection()

    def set_tasks(self, tasks: list[TaskDefinition]) -> None:
        self._tasks = list(tasks)
        self.graph_widget.set_tasks(self._tasks)
        self._refresh_selection_details()

    def set_primary_workspace_index(
        self,
        index: int,
        *,
        preserve_multi_selection: bool,
        emit_signal: bool,
    ) -> None:
        self.graph_widget.set_primary_workspace_index(
            index,
            preserve_multi_selection=preserve_multi_selection,
            emit_signal=emit_signal,
        )
        self._refresh_selection_details()

    def toggle_workspace_index_selection(self, index: int, *, emit_signal: bool) -> None:
        self.graph_widget.toggle_workspace_index_selection(index, emit_signal=emit_signal)
        self._refresh_selection_details()

    def selected_task(self) -> TaskDefinition | None:
        workspace_index = self.graph_widget.primary_workspace_index()
        if workspace_index is None or workspace_index >= len(self._tasks):
            return None
        return self._tasks[workspace_index]

    def selected_tasks(self) -> list[TaskDefinition]:
        selected_indexes = self.graph_widget.selected_workspace_indexes()
        return [self._tasks[index] for index in selected_indexes if 0 <= index < len(self._tasks)]

    def selected_workspace_indexes(self) -> list[int]:
        return self.graph_widget.selected_workspace_indexes()

    def primary_workspace_index(self) -> int | None:
        return self.graph_widget.primary_workspace_index()

    def _on_graph_selection_changed(self, primary_node: _TaskNode | None, selected_nodes: list[_TaskNode]) -> None:
        self._refresh_selection_details()
        primary_task = primary_node.task if primary_node is not None else None
        selected_tasks = [node.task for node in selected_nodes]
        self.selection_changed.emit(primary_task, selected_tasks)

    def _refresh_selection_details(self) -> None:
        task = self.selected_task()
        selected_tasks = self.selected_tasks()
        if task is None:
            self._render_empty_selection()
            return

        self.training_task_label.setText(f"Training task: {task.name}")
        self.selection_label.setText(f"Selected tasks: {len(selected_tasks)}")

        self.selection_list.clear()
        for selected_task in selected_tasks:
            label = selected_task.name
            if isinstance(selected_task, DerivedTaskDefinition):
                label = f"{label} (derived)"
            self.selection_list.addItem(QListWidgetItem(label))

        lines = [
            "Selected task",
            "",
            f"Name: {task.name}",
            f"Environment: {task.environment_id}",
            f"Task ID: {task.task_id or 'none'}",
            f"Kind: {'derived task' if isinstance(task, DerivedTaskDefinition) else 'task'}",
        ]

        if isinstance(task, DerivedTaskDefinition):
            lines.extend(
                [
                    f"Parent task id: {task.parent_task_id or 'none'}",
                    f"Derivation reason: {task.derivation_reason or 'none'}",
                    f"Source episode: {task.source_episode_id if task.source_episode_id is not None else 'none'}",
                    f"Source moment: {task.source_moment_index if task.source_moment_index is not None else 'none'}",
                    f"Source run: {task.source_run_id or 'none'}",
                ]
            )

        lines.extend(
            [
                "",
                "Config:",
                pformat(task.config, width=72),
                "",
                "Reward config:",
                pformat(task.reward_config, width=72),
                "",
                "Termination config:",
                pformat(task.termination_config, width=72),
                "",
                "Metadata:",
                pformat(task.metadata, width=72),
            ]
        )
        self.task_details.setPlainText("\n".join(lines))

    def _render_empty_selection(self) -> None:
        self.training_task_label.setText("Training task: -")
        self.selection_label.setText(f"Selected tasks: {len(self.selected_tasks())}")
        self.task_details.setPlainText("Select a task node to inspect it.")
        self.selection_list.clear()
