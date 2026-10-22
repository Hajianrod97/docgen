from dataclasses import asdict, dataclass


@dataclass
class Task:
    id: int
    title: str
    done: bool = False


class TaskStore:
    def __init__(self):
        self._tasks: dict[int, Task] = {}
        self._next_id = 1

    def all(self) -> list[dict]:
        return [asdict(t) for t in self._tasks.values()]

    def add(self, title: str) -> dict:
        task = Task(id=self._next_id, title=title.strip())
        self._tasks[task.id] = task
        self._next_id += 1
        return asdict(task)

    def complete(self, task_id: int) -> dict | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        task.done = True
        return asdict(task)
