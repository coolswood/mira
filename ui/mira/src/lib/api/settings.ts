import { fetchJson, putJson } from "./http"

// One parallel-comparison model row: reviews every PR as a shadow pass.
export type CompareModelEntry = {
  provider: string // catalog backend; "" = active provider
  model: string
  reasoning_effort: string // "off" | backend-supported level
}

export type CompareProviderOptions = {
  backend: string
  label: string
  options: { value: string; label: string; recommended?: boolean }[]
  effort_levels: { value: string; label: string; recommended?: boolean }[]
}

// Model selection, cost estimate, and admin review-config overrides.
export type ModelsSettings = {
  indexing_model: string
  review_model: string
  security_model: string
  indexing_thinking_mode: string
  review_thinking_mode: string
  security_thinking_mode: string
  api_style: string
  compare_models: CompareModelEntry[]
}

export type ModelsSettingsResponse = {
  indexing_model: string
  review_model: string
  security_model: string
  backend: string
  indexing_source: "dashboard" | "config"
  review_source: "dashboard" | "config"
  security_source: "dashboard" | "config"
  config_indexing_model: string
  config_review_model: string
  config_security_model: string
  indexing_options: {
    value: string
    label: string
    recommended?: boolean
  }[]
  review_options: { value: string; label: string; recommended?: boolean }[]
  security_options: { value: string; label: string; recommended?: boolean }[]
  indexing_thinking_mode: string
  review_thinking_mode: string
  security_thinking_mode: string
  thinking_options: {
    value: string
    label: string
    recommended?: boolean
  }[]
  effort_hint: string
  api_style: string
  api_style_options: {
    value: string
    label: string
    recommended?: boolean
  }[]
  compare_models: CompareModelEntry[]
  compare_source: "dashboard" | "config"
  compare_providers: CompareProviderOptions[]
  compare_max: number
}

export const settingsApi = {
  getModels: () => fetchJson<ModelsSettingsResponse>("/api/settings/models"),

  saveModels: (body: ModelsSettings) =>
    putJson<{ ok: boolean }>("/api/settings/models", body),

  getCostEstimate: () =>
    fetchJson<{
      estimated_usd: number
      input_tokens: number
      output_tokens: number
      model: string
      file_count: number
    }>("/api/indexing/estimate"),

  getGlobalSettings: () =>
    fetchJson<{
      overrides: {
        filter?: Record<string, number | boolean | string>
        review?: Record<string, number | boolean | string>
      }
      effective: Record<string, unknown>
    }>("/api/admin/settings"),

  saveGlobalSettings: (
    overrides: Record<string, Record<string, number | boolean | string>>
  ) => putJson<{ ok: boolean }>("/api/admin/settings", { overrides }),
}
