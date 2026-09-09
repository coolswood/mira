import { ArrowLeft, ExternalLink, GitCompare } from "lucide-react"
import { useEffect, useState } from "react"
import { Link, useParams } from "react-router"

import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { activityApi } from "@/lib/api/activity"
import type {
  ComparePassModel,
  CompareRoundModel,
} from "@/lib/api/types"
import { useDocumentTitle } from "@/lib/hooks"

// Side-by-side model comparison for one PR. Passes are grouped into rounds
// by the reviewed head SHA; each compare column carries the model's own
// findings plus its overlap with the round's main review.

const SEV_DOT: Record<string, string> = {
  blocker: "bg-destructive",
  warning: "bg-amber-500",
  suggestion: "bg-sky-500",
  nitpick: "bg-sky-500",
  // Comments recorded before severity was stringified land here as ints.
  "4": "bg-destructive",
  "3": "bg-amber-500",
  "2": "bg-sky-500",
  "1": "bg-sky-500",
}

function severityKey(raw: string): string {
  const v = (raw || "").toLowerCase()
  return SEV_DOT[v] ? v : "suggestion"
}

function formatDuration(ms: number): string {
  if (!ms) return "—"
  const s = Math.round(ms / 1000)
  return s >= 90 ? `${Math.round(s / 60)}m` : `${s}s`
}

function formatTokens(n: number): string {
  if (!n) return "—"
  return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n)
}

function relativeTime(ts: number): string {
  const diff = Math.max(0, Date.now() / 1000 - ts)
  if (diff < 3600) return `${Math.max(1, Math.round(diff / 60))}m ago`
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`
  return `${Math.round(diff / 86400)}d ago`
}

function PassColumn({ pass }: { pass: ComparePassModel }) {
  const isMain = pass.kind !== "compare"
  return (
    <Card className="min-w-[300px] flex-1 basis-0">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="truncate font-mono text-sm">
            {pass.model || (isMain ? "Main review" : "shadow")}
          </CardTitle>
          <Badge variant={isMain ? "default" : "secondary"} className="shrink-0">
            {isMain ? "Main" : "Compare"}
          </Badge>
        </div>
        <CardDescription className="text-xs">
          {pass.status === "failed"
            ? `Failed — ${pass.error || "error"}`
            : `${pass.blockers} blocker · ${pass.warnings} warning · ${pass.suggestions} suggestion`}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          <span>{pass.comments.length} findings</span>
          <span>{pass.files_reviewed} files</span>
          <span>{formatTokens(pass.tokens_used)} tokens</span>
          <span>{formatDuration(pass.duration_ms)}</span>
          <span>{relativeTime(pass.created_at)}</span>
        </div>
        {pass.comments.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            No findings recorded for this pass.
          </p>
        ) : (
          <ul className="space-y-2">
            {pass.comments.map((c) => (
              <li key={c.id} className="space-y-0.5 rounded-md border p-2 text-xs">
                <div className="flex items-start gap-2">
                  <span
                    aria-hidden
                    className={`mt-1 h-2 w-2 shrink-0 rounded-full ${
                      SEV_DOT[severityKey(c.severity)]
                    }`}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="font-medium leading-snug">{c.title}</p>
                    <p className="truncate font-mono text-[11px] text-muted-foreground">
                      {c.path}:{c.line}
                      {c.category ? ` · ${c.category}` : ""}
                    </p>
                  </div>
                </div>
                {c.body && (
                  <details className="ml-4 text-muted-foreground">
                    <summary className="cursor-pointer select-none">Details</summary>
                    <p className="mt-1 whitespace-pre-wrap">{c.body}</p>
                  </details>
                )}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  )
}

function RoundCard({ round }: { round: CompareRoundModel }) {
  const sha = round.head_sha.startsWith("legacy:")
    ? round.head_sha
    : round.head_sha.slice(0, 8)
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Round @ {sha}</CardTitle>
        <CardDescription>{relativeTime(round.created_at)}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {round.overlaps.length > 0 && (
          <div className="overflow-x-auto rounded-md border">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b bg-muted/50 text-left font-medium">
                  <th className="p-2">Model</th>
                  <th className="p-2">Both found</th>
                  <th className="p-2">Only main</th>
                  <th className="p-2">Only this model</th>
                </tr>
              </thead>
              <tbody>
                {round.overlaps.map((o) => (
                  <tr key={o.model} className="border-b last:border-0">
                    <td className="p-2 font-mono">{o.model}</td>
                    <td className="p-2">{o.shared}</td>
                    <td className="p-2">{o.only_main}</td>
                    <td className="p-2">{o.only_compare}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="flex flex-wrap gap-4">
          {round.passes.map((p) => (
            <PassColumn key={p.review_id} pass={p} />
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function ComparePage() {
  useDocumentTitle("Model comparison")
  const { owner, repo, prNumber } = useParams()
  const [rounds, setRounds] = useState<CompareRoundModel[] | null>(null)
  const [header, setHeader] = useState<{
    title: string
    url: string
  } | null>(null)
  const [error, setError] = useState("")

  useEffect(() => {
    if (!owner || !repo || !prNumber) return
    setRounds(null)
    setError("")
    activityApi
      .getCompareDetail(owner, repo, Number(prNumber))
      .then((d) => {
        setHeader({ title: d.pr_title, url: d.pr_url })
        setRounds(d.rounds)
      })
      .catch((err) =>
        setError(err instanceof Error ? err.message : String(err))
      )
  }, [owner, repo, prNumber])

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 space-y-1">
          <div className="flex items-center gap-2 text-sm">
            <Link
              to="/activity"
              className="inline-flex items-center gap-1 text-muted-foreground transition-colors hover:text-foreground"
            >
              <ArrowLeft className="h-4 w-4" />
              Activity
            </Link>
          </div>
          <h1 className="flex min-w-0 items-center gap-2 text-2xl font-semibold tracking-tight">
            <GitCompare className="h-5 w-5 shrink-0 text-muted-foreground" />
            <span className="truncate">
              #{prNumber} {header?.title ?? ""}
            </span>
            {header?.url && (
              <a
                href={header.url}
                target="_blank"
                rel="noreferrer"
                aria-label="Open PR on GitHub"
                className="shrink-0 text-muted-foreground transition-colors hover:text-foreground"
              >
                <ExternalLink className="h-4 w-4" />
              </a>
            )}
          </h1>
          <p className="text-sm text-muted-foreground">
            {owner}/{repo} — review passes grouped by the head SHA they
            reviewed; shadow passes never post to GitHub.
          </p>
        </div>
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}
      {!error && rounds === null && (
        <div className="space-y-4">
          <Skeleton className="h-48 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      )}
      {rounds !== null && rounds.length === 0 && (
        <p className="text-sm text-muted-foreground">
          No review passes recorded for this PR.
        </p>
      )}
      {rounds?.map((round, i) => (
        <RoundCard key={round.head_sha + i} round={round} />
      ))}

      {rounds !== null && rounds.length > 0 && (
        <p className="text-xs text-muted-foreground">
          To compare models, add them under Settings → Models → Parallel review
          models; every PR is then reviewed by all of them side by side.
        </p>
      )}
    </div>
  )
}
