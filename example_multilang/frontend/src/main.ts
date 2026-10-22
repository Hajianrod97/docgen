import { addTask, completeTask, fetchTasks, Task } from "./api";

const list = document.querySelector<HTMLUListElement>("#tasks")!;
const form = document.querySelector<HTMLFormElement>("#new-task")!;

function render(tasks: Task[]) {
  list.innerHTML = "";
  for (const task of tasks) {
    const li = document.createElement("li");
    li.textContent = task.title;
    li.className = task.done ? "done" : "";
    li.onclick = async () => {
      await completeTask(task.id);
      render(await fetchTasks());
    };
    list.appendChild(li);
  }
}

form.onsubmit = async (event) => {
  event.preventDefault();
  const input = form.querySelector("input")!;
  await addTask(input.value);
  input.value = "";
  render(await fetchTasks());
};

fetchTasks().then(render);
