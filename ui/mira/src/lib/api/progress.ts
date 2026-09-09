import { fetchJson } from "./http"

// Live progress for long-running jobs (PR reviews, indexing). Backed by the
// in-memory ProgressTracker: `GET /api/progress` returns the snapshot, and the
// `job_progress` SSE event on `/api/events` pushes updates as they happen.

export interface JobProgressEvent {
  ts: number
  kind: string
  message: string
  data: Record<string, unknown>
}

export interface JobProgress {
  key: string
  kind: "review" | "indexing"
  repo: string
  pr_number: number
  pr_title: string
  pr_url: string
  status: "running" | "completed" | "failed" | "cancelled"
  stage: string
  review_round: number
  files_total: number
  files_done: number
  chunks_total: number
  chunks_done: number
  calls_started: number
  calls_in_flight: number
  calls_failed: number
  calls_by_stage: Record<string, number>
  tokens_input: number
  tokens_cached: number
  tokens_output: number
  tokens_reasoning: number
  items: Record<string, number>
  started_at: number
  updated_at: number
  finished_at: number
  elapsed_s: number
  eta_s: number | null
  error: string
  events: JobProgressEvent[]
}

export const progressApi = {
  getProgress: (activeOnly = false) =>
    fetchJson<JobProgress[]>(`/api/progress${activeOnly ? "?active_only=true" : ""}`),
}
