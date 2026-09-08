import { useEffect, useState } from "react"

import { api } from "@/lib/api"
import type { JobProgress, JobProgressEvent } from "@/lib/api/progress"

const API_BASE = import.meta.env.VITE_API_URL || ""

const MAX_EVENTS_PER_JOB = 8

/**
 * Live progress for long-running jobs (reviews, indexing).
 *
 * One initial `GET /api/progress` snapshot, then live updates via the
 * `job_progress` SSE event on the existing `/api/events` stream — no polling
 * while the tab is open. A slow 30s re-sync covers dropped SSE connections.
 */
export function useJobProgress() {
  const [jobs, setJobs] = useState<JobProgress[]>([])
  const [recentEvents, setRecentEvents] = useState<
    Record<string, JobProgressEvent[]>
  >({})

  useEffect(() => {
    let cancelled = false

    const load = () => {
      api
        .getProgress()
        .then((snapshot) => {
          if (cancelled) return
          setJobs(snapshot)
          // Seed the per-job feeds from the tracker's stored events; SSE
          // events append on top as they arrive.
          setRecentEvents((prev) => {
            const next = { ...prev }
            for (const job of snapshot) {
              if (!next[job.key]) {
                next[job.key] = job.events.slice(-MAX_EVENTS_PER_JOB)
              }
            }
            return next
          })
        })
        .catch(() => {})
    }

    load()
    const interval = setInterval(load, 30000)

    const eventSource = new EventSource(`${API_BASE}/api/events`, {
      withCredentials: true,
    })

    eventSource.addEventListener("job_progress", (e) => {
      try {
        const payload = JSON.parse((e as MessageEvent).data) as {
          job: JobProgress
          event: JobProgressEvent
        }
        if (cancelled) return
        setJobs((prev) => {
          const index = prev.findIndex((j) => j.key === payload.job.key)
          if (index === -1) return [...prev, payload.job]
          const next = [...prev]
          next[index] = { ...next[index], ...payload.job }
          return next
        })
        setRecentEvents((prev) => {
          const feed = prev[payload.job.key] ?? []
          return {
            ...prev,
            [payload.job.key]: [...feed, payload.event].slice(
              -MAX_EVENTS_PER_JOB
            ),
          }
        })
      } catch {
        // ignore malformed events
      }
    })

    return () => {
      cancelled = true
      clearInterval(interval)
      eventSource.close()
    }
  }, [])

  return { jobs, recentEvents }
}
