import { useEffect, useState } from "react"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { JobProgress, JobProgressEvent } from "@/lib/api/progress"
import { useJobProgress } from "@/lib/use-progress"

/** Ticks once a second so running jobs never look frozen between
 * SSE pushes. `null` pauses the timer (no running jobs). */
function useNowSeconds(active: boolean) {
  const [now, setNow] = useState(() => Date.now() / 1000)
  useEffect(() => {
    if (!active) return
    const interval = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(interval)
  }, [active])
  return now
}

function formatDuration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return `${n}`
}

const STAGE_LABELS: Record<string, string> = {
  starting: "Starting",
  walkthrough: "Walkthrough",
  review: "Reviewing chunks",
  critique: "Self-critique",
  summary: "Summary",
  posting: "Posting to GitHub",
  fetching: "Fetching repo",
  files: "Summarizing files",
  directories: "Directory summaries",
}

function stageLabel(stage: string): string {
  return STAGE_LABELS[stage] ?? stage
}

function JobRow({
  job,
  lastEvent,
  now,
}: {
  job: JobProgress
  lastEvent?: JobProgressEvent
  now: number
}) {
  const chunkPct =
    job.chunks_total > 0
      ? Math.min(100, Math.round((job.chunks_done / job.chunks_total) * 100))
      : null
  const filePct =
    job.files_total > 0
      ? Math.min(100, Math.round((job.files_done / job.files_total) * 100))
      : null
  const pct = chunkPct ?? filePct
  // Prefer the client-side clock so elapsed time ticks smoothly even when
  // the tracker is quiet between events.
  const elapsed =
    job.status === "running" ? Math.max(0, now - job.started_at) : job.elapsed_s

  return (
    <div className="space-y-2 py-3 first:pt-0 last:pb-0">
      <div className="flex items-baseline justify-between gap-2">
        <div className="min-w-0">
          <a
            href={job.pr_url || undefined}
            target="_blank"
            rel="noreferrer"
            className="truncate text-sm font-medium hover:underline"
          >
            {job.kind === "indexing" ? (
              <>Indexing {job.repo}</>
            ) : (
              <>
                {job.kind === "compare" && (
                  <span className="mr-1 rounded bg-muted px-1 py-0.5 font-mono text-[11px] font-normal">
                    {job.key.split(":cmp:").pop()}
                  </span>
                )}
                #{job.pr_number} {job.pr_title || job.repo}
              </>
            )}
          </a>
          <div className="text-xs text-muted-foreground">
            {job.kind !== "indexing" && `${job.repo} · `}
            {stageLabel(job.stage)}
            {job.review_round > 1 ? ` (round ${job.review_round})` : ""}
            {job.calls_in_flight > 0
              ? ` · ${job.calls_in_flight} LLM call${job.calls_in_flight > 1 ? "s" : ""} running`
              : ""}
          </div>
        </div>
        <div className="shrink-0 text-right text-xs tabular-nums text-muted-foreground">
          {job.status === "running" ? (
            <>
              {formatDuration(elapsed)}
              {job.eta_s != null ? ` / ~${formatDuration(job.eta_s)} left` : ""}
            </>
          ) : (
            job.status
          )}
        </div>
      </div>

      {pct !== null && (
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      )}
      {chunkPct !== null && (
        <div className="text-xs text-muted-foreground">
          chunk {Math.min(job.chunks_done + 1, job.chunks_total)}/{job.chunks_total}
          {job.files_total > 0 ? ` · ${job.files_total} files` : ""}
        </div>
      )}
      {filePct !== null && (
        <div className="text-xs text-muted-foreground">
          {job.files_done}/{job.files_total} files
        </div>
      )}

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground tabular-nums">
        <span>
          tokens: {formatTokens(job.tokens_input)} in
          {job.tokens_cached > 0 ? ` (${formatTokens(job.tokens_cached)} cached)` : ""} /{" "}
          {formatTokens(job.tokens_output)} out
        </span>
        <span>
          calls: {job.calls_started}
          {job.calls_failed > 0 ? ` (${job.calls_failed} failed)` : ""}
        </span>
      </div>

      {lastEvent && (
        <div className="truncate text-xs text-muted-foreground" title={lastEvent.message}>
          {lastEvent.message}
        </div>
      )}
      {job.error && (
        <div className="truncate text-xs text-destructive" title={job.error}>
          {job.error}
        </div>
      )}
    </div>
  )
}

/**
 * Live view of running jobs (reviews + indexing). Shown on the dashboard and,
 * filtered by repo, on the repo detail page. Renders nothing when idle so
 * quiet periods stay quiet.
 */
export function JobProgressCard({
  repo,
  showRecent = false,
}: {
  repo?: string
  showRecent?: boolean
}) {
  const { jobs, recentEvents } = useJobProgress()
  const now = useNowSeconds(jobs.some((j) => j.status === "running"))

  const running = jobs.filter(
    (j) => j.status === "running" && (!repo || j.repo === repo)
  )
  const recent = showRecent
    ? jobs
        .filter(
          (j) =>
            j.status !== "running" &&
            (!repo || j.repo === repo) &&
            Date.now() / 1000 - j.finished_at < 3600
        )
        .slice(-3)
    : []

  if (running.length === 0 && recent.length === 0) return null

  return (
    <Card className="relative overflow-hidden border-primary/30 bg-primary/5">
      {running.length > 0 && (
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary to-transparent [background-size:200%_100%] [animation:shimmer_2s_linear_infinite]" />
      )}
      <CardHeader className="pb-2">
        <CardTitle className="text-base">
          {running.length > 0
            ? repo
              ? "Working on this repository"
              : "In progress"
            : "Recent activity"}
        </CardTitle>
        {running.length > 0 && (
          <CardDescription>
            {running.length === 1
              ? "1 job running — live"
              : `${running.length} jobs running — live`}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent className="divide-y">
        {running.map((job) => (
          <JobRow
            key={job.key}
            job={job}
            lastEvent={recentEvents[job.key]?.at(-1)}
            now={now}
          />
        ))}
        {recent.map((job) => (
          <JobRow key={job.key} job={job} now={now} />
        ))}
      </CardContent>
    </Card>
  )
}
