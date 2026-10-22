import { config } from "./config";

export interface Task {
  id: number;
  title: string;
  done: boolean;
}

export async function fetchTasks(): Promise<Task[]> {
  const res = await fetch(`${config.apiUrl}/tasks`);
  return res.json();
}

export async function addTask(title: string): Promise<Task> {
  const res = await fetch(`${config.apiUrl}/tasks?title=${encodeURIComponent(title)}`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function completeTask(id: number): Promise<Task> {
  const res = await fetch(`${config.apiUrl}/tasks/${id}/done`, { method: "POST" });
  return res.json();
}
