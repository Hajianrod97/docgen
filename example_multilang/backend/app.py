from fastapi import FastAPI, HTTPException

from backend.store import TaskStore

app = FastAPI()
store = TaskStore()


@app.get("/tasks")
def list_tasks():
    return store.all()


@app.post("/tasks")
def add_task(title: str):
    if not title.strip():
        raise HTTPException(400, "title must not be empty")
    return store.add(title)


@app.post("/tasks/{task_id}/done")
def complete_task(task_id: int):
    task = store.complete(task_id)
    if task is None:
        raise HTTPException(404, "no such task")
    return task
