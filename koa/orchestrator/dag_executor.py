"""DAG Executor - Topological sort and result tracking for multi-intent execution.

Provides utilities for ordering sub-tasks by their dependencies and
tracking execution results.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .intent_analyzer import SubTask

logger = logging.getLogger(__name__)


@dataclass
class SubTaskResult:
    """Result of a single sub-task execution."""

    sub_task_id: int
    description: str
    response: str
    status: str  # "completed" or "error"
    duration_ms: int = 0
    token_usage: Dict[str, int] = field(default_factory=dict)


def get_runnable_tasks(
    level: List[SubTask],
    prior_results: Dict[int, SubTaskResult],
) -> tuple[List[SubTask], List[SubTask]]:
    """Split a level into runnable and skipped tasks.

    A task is skipped if any of its dependencies have status != "completed".

    Returns:
        (runnable, skipped) tuple
    """
    runnable: List[SubTask] = []
    skipped: List[SubTask] = []
    for task in level:
        deps_ok = all(
            prior_results.get(dep_id) is not None
            and prior_results[dep_id].status == "completed"
            for dep_id in task.depends_on
        )
        if deps_ok:
            runnable.append(task)
        else:
            skipped.append(task)
    return runnable, skipped


def aggregate_token_usage(results: Dict[int, SubTaskResult]) -> Dict[str, int]:
    """Aggregate token usage across all sub-task results.

    Keys match the ReAct loop's EXECUTION_END format: input_tokens / output_tokens.
    """
    total_input = 0
    total_output = 0
    for result in results.values():
        total_input += result.token_usage.get("input_tokens", 0)
        total_output += result.token_usage.get("output_tokens", 0)
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
    }


def topological_sort(sub_tasks: List[SubTask]) -> List[List[SubTask]]:
    """Sort sub-tasks into parallel execution levels.

    Returns a list of levels. Tasks within the same level have all their
    dependencies satisfied by prior levels and can execute in parallel.

    Raises:
        ValueError: If a dependency cycle is detected.
    """
    if not sub_tasks:
        return []

    id_to_task = {st.id: st for st in sub_tasks}
    in_degree = {st.id: 0 for st in sub_tasks}
    dependents: Dict[int, List[int]] = defaultdict(list)

    for st in sub_tasks:
        for dep_id in st.depends_on:
            if dep_id in id_to_task:
                in_degree[st.id] += 1
                dependents[dep_id].append(st.id)

    levels: List[List[SubTask]] = []
    remaining = set(id_to_task.keys())

    while remaining:
        # Collect all tasks with in_degree 0
        ready_ids = [sid for sid in remaining if in_degree[sid] == 0]
        if not ready_ids:
            raise ValueError(
                f"Cycle detected in sub-task dependencies. "
                f"Remaining tasks: {remaining}"
            )

        level = [id_to_task[sid] for sid in sorted(ready_ids)]
        levels.append(level)

        for sid in ready_ids:
            remaining.remove(sid)
            for dep_id in dependents[sid]:
                in_degree[dep_id] -= 1

    return levels
