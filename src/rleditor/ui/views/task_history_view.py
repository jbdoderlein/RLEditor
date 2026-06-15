from __future__ import annotations

from pprint import pformat

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rleditor.core.models import DerivedTaskDefinition, TaskDefinition


_WORKSPACE_INDEX_ROLE = int(Qt.ItemDataRole.UserRole)


class TaskHistoryView(QWidget):
    selection_changed = Signal(object, object)
    create_task_requested = Signal()
    import_task_requested = Signal()
    edit_task_requested = Signal(int)
    copy_task_requested = Signal(int)
    delete_tasks_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._tasks: list[TaskDefinition] = []
        self._item_by_workspace_index: dict[int, QTreeWidgetItem] = {}

        root = QVBoxLayout(self)

        title = QLabel("Task History")
        title.setObjectName("TitleLabel")
        self.import_task_button = QPushButton("Import Task", self)
        self.import_task_button.setToolTip("Import a task JSON or a generated curriculum task file.")
        self.import_task_button.clicked.connect(self.import_task_requested.emit)
        add_task_button = QPushButton("Add New Task", self)
        add_task_button.clicked.connect(self.create_task_requested.emit)

        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.import_task_button)
        title_row.addWidget(add_task_button)

        subtitle = QLabel(
            "Browse tasks as a tree, choose the primary training task, and Ctrl-click for parallel training."
        )
        subtitle.setObjectName("SubtitleLabel")
        subtitle.setWordWrap(True)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        left_panel = QWidget(splitter)
        left_layout = QVBoxLayout(left_panel)
        left_group = QGroupBox("Task Browser", left_panel)
        left_group_layout = QVBoxLayout(left_group)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Task", "Kind", "ID"])
        self.tree_widget.setRootIsDecorated(True)
        self.tree_widget.setAlternatingRowColors(True)
        self.tree_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree_widget.setMinimumWidth(260)
        left_group_layout.addWidget(self.tree_widget)
        left_layout.addWidget(left_group)

        right_panel = QWidget(splitter)
        right_layout = QVBoxLayout(right_panel)

        self.training_task_label = QLabel("Training task: -")
        self.training_task_label.setWordWrap(True)
        self.selection_label = QLabel("Selected tasks: 0")
        self.selection_label.setWordWrap(True)

        actions_layout = QHBoxLayout()
        self.edit_task_button = QPushButton("Edit Task", right_panel)
        self.edit_task_button.setToolTip("Open the selected task in the Task Editor tab.")
        self.edit_task_button.setEnabled(False)
        self.copy_task_button = QPushButton("Copy Task", right_panel)
        self.copy_task_button.setToolTip("Duplicate the selected task.")
        self.copy_task_button.setEnabled(False)
        self.delete_task_button = QPushButton("Delete Task", right_panel)
        self.delete_task_button.setToolTip("Delete the selected task(s) from the workspace.")
        self.delete_task_button.setEnabled(False)
        actions_layout.addWidget(self.edit_task_button)
        actions_layout.addWidget(self.copy_task_button)
        actions_layout.addWidget(self.delete_task_button)

        details_group = QGroupBox("Task Details", right_panel)
        details_layout = QFormLayout(details_group)
        self.task_details = QTextEdit(details_group)
        self.task_details.setReadOnly(True)
        self.task_details.setMinimumHeight(100)
        details_layout.addRow(self.task_details)

        selection_group = QGroupBox("Training Selection", right_panel)
        selection_layout = QVBoxLayout(selection_group)
        self.selection_list = QListWidget(selection_group)
        selection_layout.addWidget(self.selection_list)

        right_layout.addWidget(self.training_task_label)
        right_layout.addWidget(self.selection_label)
        right_layout.addLayout(actions_layout)
        right_layout.addWidget(details_group)
        right_layout.addWidget(selection_group, 1)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([520, 520])

        root.addLayout(title_row)
        root.addWidget(subtitle)
        root.addWidget(splitter, 1)

        self.tree_widget.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.tree_widget.currentItemChanged.connect(self._on_current_item_changed)
        self.edit_task_button.clicked.connect(self._emit_edit_selected_task)
        self.copy_task_button.clicked.connect(self._emit_copy_selected_task)
        self.delete_task_button.clicked.connect(self._emit_delete_selected_tasks)

        self._render_empty_selection()

    def set_tasks(self, tasks: list[TaskDefinition]) -> None:
        previous_primary = self.primary_workspace_index()
        previous_selected = set(self.selected_workspace_indexes())
        self._tasks = list(tasks)
        self._rebuild_tree()

        if previous_primary is not None and previous_primary < len(self._tasks):
            primary_index = previous_primary
        else:
            primary_index = 0 if self._tasks else None

        selected_indexes = {
            index for index in previous_selected if 0 <= index < len(self._tasks)
        }
        if primary_index is not None:
            selected_indexes.add(primary_index)

        self._apply_selection(
            primary_index=primary_index,
            selected_indexes=selected_indexes,
            emit_signal=False,
        )
        self._refresh_selection_details()

    def set_primary_workspace_index(
        self,
        index: int,
        *,
        preserve_multi_selection: bool,
        emit_signal: bool,
    ) -> None:
        if index < 0 or index >= len(self._tasks):
            return
        selected_indexes = set(self.selected_workspace_indexes()) if preserve_multi_selection else set()
        selected_indexes.add(index)
        self._apply_selection(
            primary_index=index,
            selected_indexes=selected_indexes,
            emit_signal=emit_signal,
        )
        self._refresh_selection_details()

    def toggle_workspace_index_selection(self, index: int, *, emit_signal: bool) -> None:
        item = self._item_by_workspace_index.get(index)
        if item is None:
            return

        selected_indexes = set(self.selected_workspace_indexes())
        if index in selected_indexes:
            if len(selected_indexes) <= 1:
                primary_index = index
            else:
                selected_indexes.remove(index)
                primary_index = min(selected_indexes)
        else:
            selected_indexes.add(index)
            primary_index = index

        self._apply_selection(
            primary_index=primary_index,
            selected_indexes=selected_indexes,
            emit_signal=emit_signal,
        )
        self._refresh_selection_details()

    def selected_task(self) -> TaskDefinition | None:
        workspace_index = self.primary_workspace_index()
        if workspace_index is None or workspace_index >= len(self._tasks):
            return None
        return self._tasks[workspace_index]

    def selected_tasks(self) -> list[TaskDefinition]:
        selected_indexes = self.selected_workspace_indexes()
        return [self._tasks[index] for index in selected_indexes if 0 <= index < len(self._tasks)]

    def selected_workspace_indexes(self) -> list[int]:
        indexes = [
            self._workspace_index(item)
            for item in self.tree_widget.selectedItems()
        ]
        return sorted(index for index in indexes if index is not None)

    def primary_workspace_index(self) -> int | None:
        current_item = self.tree_widget.currentItem()
        if current_item is not None and current_item.isSelected():
            return self._workspace_index(current_item)
        selected_indexes = self.selected_workspace_indexes()
        return selected_indexes[0] if selected_indexes else None

    def _rebuild_tree(self) -> None:
        self.tree_widget.blockSignals(True)
        self.tree_widget.clear()
        self._item_by_workspace_index.clear()

        identifier_to_index: dict[str, int] = {}
        for index, task in enumerate(self._tasks):
            if task.task_id:
                identifier_to_index[str(task.task_id)] = index
            identifier_to_index.setdefault(task.name, index)

        children: dict[int | None, list[int]] = {}
        parent_by_index: dict[int, int | None] = {}
        for index, task in enumerate(self._tasks):
            parent_index = self._parent_workspace_index(task, identifier_to_index)
            if parent_index == index:
                parent_index = None
            parent_by_index[index] = parent_index
            children.setdefault(parent_index, []).append(index)

        for child_indexes in children.values():
            child_indexes.sort(key=lambda child_index: self._tasks[child_index].name.lower())

        visited: set[int] = set()

        def add_item(task_index: int, parent_item: QTreeWidgetItem | None) -> None:
            if task_index in visited:
                return
            visited.add(task_index)
            task = self._tasks[task_index]
            item = QTreeWidgetItem(
                [
                    task.name,
                    "Derived" if isinstance(task, DerivedTaskDefinition) else "Task",
                    task.task_id or "-",
                ]
            )
            item.setData(0, _WORKSPACE_INDEX_ROLE, task_index)
            item.setToolTip(0, task.name)
            self._item_by_workspace_index[task_index] = item
            if parent_item is None:
                self.tree_widget.addTopLevelItem(item)
            else:
                parent_item.addChild(item)

            for child_index in children.get(task_index, []):
                add_item(child_index, item)
            item.setExpanded(True)

        for root_index in children.get(None, []):
            add_item(root_index, None)

        for task_index in range(len(self._tasks)):
            if task_index not in visited:
                add_item(task_index, None)

        self.tree_widget.resizeColumnToContents(0)
        self.tree_widget.blockSignals(False)

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

    def _apply_selection(
        self,
        *,
        primary_index: int | None,
        selected_indexes: set[int],
        emit_signal: bool,
    ) -> None:
        self.tree_widget.blockSignals(True)
        self.tree_widget.clearSelection()
        primary_item = self._item_by_workspace_index.get(primary_index) if primary_index is not None else None
        if primary_item is not None:
            self.tree_widget.setCurrentItem(primary_item)
        for selected_index in sorted(selected_indexes):
            item = self._item_by_workspace_index.get(selected_index)
            if item is not None:
                item.setSelected(True)

        if primary_item is not None:
            primary_item.setSelected(True)
            self.tree_widget.scrollToItem(primary_item)
        self.tree_widget.blockSignals(False)

        if emit_signal:
            self._emit_selection_changed()

    def _on_tree_selection_changed(self) -> None:
        self._refresh_selection_details()
        self._emit_selection_changed()

    def _on_current_item_changed(self, current: QTreeWidgetItem | None, previous: QTreeWidgetItem | None) -> None:
        _ = previous
        if current is not None and not current.isSelected():
            current.setSelected(True)
        self._refresh_selection_details()

    def _emit_selection_changed(self) -> None:
        self.selection_changed.emit(self.selected_task(), self.selected_tasks())

    def _refresh_selection_details(self) -> None:
        task = self.selected_task()
        selected_tasks = self.selected_tasks()
        if task is None:
            self._render_empty_selection()
            return

        self.training_task_label.setText(f"Training task: {task.name}")
        self.selection_label.setText(f"Selected tasks: {len(selected_tasks)}")
        self.edit_task_button.setEnabled(True)
        self.copy_task_button.setEnabled(True)
        self.delete_task_button.setEnabled(True)

        self.selection_list.clear()
        for selected_task in selected_tasks:
            label = selected_task.name
            if isinstance(selected_task, DerivedTaskDefinition):
                label = f"{label} (derived)"
            if selected_task is task:
                label = f"{label} [primary]"
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
        self.task_details.setPlainText("Select a task in the browser to inspect it.")
        self.selection_list.clear()
        self.edit_task_button.setEnabled(False)
        self.copy_task_button.setEnabled(False)
        self.delete_task_button.setEnabled(False)

    def _emit_edit_selected_task(self) -> None:
        index = self.primary_workspace_index()
        if index is not None:
            self.edit_task_requested.emit(index)

    def _emit_copy_selected_task(self) -> None:
        index = self.primary_workspace_index()
        if index is not None:
            self.copy_task_requested.emit(index)

    def _emit_delete_selected_tasks(self) -> None:
        indexes = self.selected_workspace_indexes()
        if indexes:
            self.delete_tasks_requested.emit(indexes)

    def _workspace_index(self, item: QTreeWidgetItem) -> int | None:
        raw_value = item.data(0, _WORKSPACE_INDEX_ROLE)
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return None
