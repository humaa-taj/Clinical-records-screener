export type Status = {
  state: 'loading' | 'ready' | 'failed';
  error: string | null;
  target_model: string;
  adapter: string | null;
  trained: boolean;
  output_order: string;
  draft_model: string;
  device: string;
  draft_device: string;
  busy: boolean;
};

export type Evidence = { id: number; text: string };

export type CriterionResult = {
  index: number;
  criterion_type: 'inclusion' | 'exclusion';
  criterion: string;
  label: string;
  adjusted?: boolean;
  valid: boolean;
  unconfirmed: boolean;
  evidence_ids: number[];
  evidence: Evidence[];
  explanation: string;
  missing_information: string[];
  errors: string[];
  raw_output: string;
  timing: Record<string, number | string>;
};

export type Overall = {
  outcome: string;
  /** Absent on reports saved before reasons were added. */
  reason?: string;
  preliminary: boolean;
  counts: Record<string, number>;
  rule: string;
};

export type DecoderRow = {
  mode: string;
  input_tokens: number;
  output_tokens: number;
  prefill_s: number;
  decode_s: number;
  total_s: number;
  tokens_per_s: number;
  tokens_match_reference: boolean;
  peak_gpu_extra_mb: number | null;
  acceptance_rate?: number;
  rounds?: number;
};

export type DecodingOutput = { rows: DecoderRow[]; draft_device: string; device: string; text: string };

export type AttentionOutput = {
  device: string;
  length: number;
  tile: number;
  reference_ms: number;
  tiled_ms: number;
  reference_gpu_extra_mb: number | null;
  tiled_gpu_extra_mb: number | null;
  max_abs_error: number;
  mean_abs_error: number;
  tiles_computed: number;
  tiles_skipped: number;
};

export type Job<T = unknown> = {
  id: string;
  kind: string;
  state: 'running' | 'done' | 'failed' | 'cancelled';
  done: number;
  total: number;
  results: CriterionResult[];
  error: string | null;
  overall: Overall | null;
  report_id: string | null;
  output: T | null;
};

export type ReportSummary = { id: string; created_utc: string; outcome: string; criteria: number; mode: string };

export type Report = {
  id: string;
  overall: Overall;
  results: CriterionResult[];
  patient_sentences: Evidence[];
};

export type Example = { note: string; inclusion: string[]; exclusion: string[]; source: string };
export type Trial = { id: string; title: string; inclusion: string[]; exclusion: string[] };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new Error('The CRS server is not responding. Check that it is still running.');
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) throw new Error(body?.message ?? 'Something went wrong. Try again.');
  return body as T;
}

const post = <T>(path: string, data: unknown) =>
  call<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });

export const api = {
  status: () => call<Status>('/api/status'),
  example: () => call<Example>('/api/example'),
  sentences: (note: string) => post<{ sentences: string[] }>('/api/sentences', { note }),
  trial: (id: string) => call<Trial>(`/api/trial/${encodeURIComponent(id)}`),
  screen: (data: { sentences: string[]; inclusion: string[]; exclusion: string[]; mode: string }) => post<{ id: string }>('/api/screen', data),
  job: <T>(id: string) => call<Job<T>>(`/api/jobs/${id}`),
  cancel: (id: string) => post<unknown>(`/api/jobs/${id}/cancel`, {}),
  compareDecoding: (data: { sentences: string[]; criterion_type: string; criterion: string; max_new_tokens: number; ignore_eos: boolean; draft_length: number }) =>
    post<{ id: string }>('/api/compare/decoding', data),
  compareAttention: (data: { length: number; tile: number }) => post<{ id: string }>('/api/compare/attention', data),
  reports: () => call<{ reports: ReportSummary[] }>('/api/reports'),
  report: (id: string) => call<Report>(`/api/reports/${id}`),
};

export const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Poll a job until it stops running, returning its final state. */
export async function waitForJob<T>(id: string, onUpdate?: (job: Job<T>) => void): Promise<Job<T>> {
  for (;;) {
    const job = await api.job<T>(id);
    onUpdate?.(job);
    if (job.state !== 'running') return job;
    await delay(400);
  }
}
