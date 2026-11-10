import { useEffect, useState, type ReactNode } from 'react';
import { api, waitForJob, type AttentionOutput, type CriterionResult, type DecodingOutput, type ReportSummary, type Status } from './api';
import { Notice } from './shared';

const TABS = [
  ['decoding', 'Decoding'],
  ['attention', 'Attention'],
  ['reports', 'Reports'],
  ['model', 'Model'],
] as const;

const MODE_NAMES: Record<string, string> = { uncached: 'No cache (reference)', cached: 'KV cache', speculative: 'Speculative' };
const number = (value: number | null | undefined, digits = 2) => (value == null ? 'n/a' : value.toFixed(digits));

function Panel({ id, tab, title, children }: { id: string; tab: string; title: string; children: ReactNode }) {
  return (
    <section id={`panel-${id}`} role="tabpanel" aria-labelledby={`tab-${id}`} hidden={tab !== id}>
      <h2 className="m-0 mb-4 text-2xl font-medium">{title}</h2>
      {children}
    </section>
  );
}

function Bars({ title, unit, rows }: { title: string; unit: string; rows: Array<{ label: string; value: number; reference: boolean }> }) {
  const max = Math.max(...rows.map((row) => row.value), 1e-9);
  return (
    <figure className="card m-0 p-6">
      <figcaption className="mb-5 text-lg">{title}</figcaption>
      <div className="flex flex-col gap-3.5">
        {rows.map((row) => (
          <div key={row.label} className="grid items-center gap-x-4 gap-y-1 sm:grid-cols-[12rem_1fr]">
            <span className="text-neutral-300">{row.label}</span>
            <div className="flex items-center gap-3">
              <div className={`h-4 rounded-sm ${row.reference ? 'hatched' : 'solid-glow'}`} style={{ width: `${(row.value / max) * 82}%`, minWidth: 3 }} />
              <span className="font-medium tabular-nums">{row.value.toFixed(2)}</span>
            </div>
          </div>
        ))}
      </div>
      <p className="m-0 mt-5 text-sm text-neutral-400">{unit}. Hatched: reference. Solid: compared method.</p>
    </figure>
  );
}

type Choice = Pick<CriterionResult, 'criterion' | 'criterion_type'>;
type LabProps = { ready: boolean; sentences: string[]; choices: Choice[]; startIndex: number; criterion: string; criterionType: string };

function DecodingLab({ ready, sentences, choices, startIndex, criterion, criterionType }: LabProps) {
  const start = choices[startIndex];
  const [picked, setPicked] = useState(startIndex);
  const [text, setText] = useState(start?.criterion ?? criterion);
  const [kind, setKind] = useState<string>(start?.criterion_type ?? criterionType);
  const [note, setNote] = useState(sentences.join('\n'));
  const [usingExample, setUsingExample] = useState(false);
  const [limit, setLimit] = useState(64);
  const [draftLength, setDraftLength] = useState(4);
  const [ignoreEos, setIgnoreEos] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [output, setOutput] = useState<DecodingOutput | null>(null);

  useEffect(() => {
    if (sentences.length || start || criterion) return;
    let active = true;
    api.example()
      .then(async (example) => {
        const split = await api.sentences(example.note);
        if (!active) return;
        setNote(split.sentences.join('\n'));
        setText(example.inclusion[0] ?? '');
        setKind('inclusion');
        setUsingExample(true);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  function pick(index: number) {
    setPicked(index);
    setText(choices[index].criterion);
    setKind(choices[index].criterion_type);
  }

  const blocked = !ready ? 'Waiting for the model to load.' : !note.trim() || !text.trim() ? 'Add patient sentences and a criterion to run this comparison.' : '';

  async function run() {
    setRunning(true);
    setError('');
    try {
      const noteLines = note.split('\n').map((line) => line.trim()).filter(Boolean);
      const { id } = await api.compareDecoding({ sentences: noteLines, criterion_type: kind, criterion: text, max_new_tokens: limit, ignore_eos: ignoreEos, draft_length: draftLength });
      const job = await waitForJob<DecodingOutput>(id);
      if (job.state === 'done' && job.output) setOutput(job.output);
      else setError(job.error ?? 'The comparison did not finish.');
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="card flex flex-col gap-4 p-5">
        {usingExample && <p className="m-0 text-sm text-neutral-300">Showing a made-up example, since there is no screening yet. You can edit it.</p>}
        {choices.length > 0 && (
          <label className="flex flex-col gap-2 text-neutral-300">
            Criterion from this result
            <select className="field" value={picked} onChange={(event) => pick(Number(event.target.value))}>
              {choices.map((choice, index) => (
                <option key={index} value={index}>
                  {index + 1}. {choice.criterion.length > 90 ? `${choice.criterion.slice(0, 90)}...` : choice.criterion}
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="flex flex-col gap-2 text-neutral-300">
          Patient sentences, one per line
          <textarea className="field min-h-28" value={note} onChange={(event) => setNote(event.target.value)} />
        </label>
        <div className="grid gap-4 sm:grid-cols-[10rem_1fr]">
          <label className="flex flex-col gap-2 text-neutral-300">
            Type
            <select className="field" value={kind} onChange={(event) => setKind(event.target.value)}>
              <option value="inclusion">Inclusion</option>
              <option value="exclusion">Exclusion</option>
            </select>
          </label>
          <label className="flex flex-col gap-2 text-neutral-300">
            Criterion
            <input className="field" value={text} onChange={(event) => setText(event.target.value)} />
          </label>
        </div>
        <div className="grid items-end gap-4 sm:grid-cols-3">
          <label className="flex flex-col gap-2 text-neutral-300">
            Maximum output tokens
            <select className="field" value={limit} onChange={(event) => setLimit(Number(event.target.value))}>
              {[16, 32, 64, 128, 256].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-2 text-neutral-300">
            Draft length
            <select className="field" value={draftLength} onChange={(event) => setDraftLength(Number(event.target.value))}>
              {[2, 4, 8].map((value) => (
                <option key={value}>{value}</option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 pb-3 text-neutral-300">
            <input type="checkbox" checked={ignoreEos} onChange={(event) => setIgnoreEos(event.target.checked)} />
            Generate the full length
          </label>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <button type="button" className="btn btn-primary" disabled={running || Boolean(blocked)} onClick={run}>
            {running ? 'Running...' : 'Run comparison'}
          </button>
          {blocked && <span className="text-sm text-neutral-300">{blocked}</span>}
        </div>
      </div>
      {error && <Notice message={error} />}
      {output && (
        <>
          <div className="card overflow-x-auto p-5">
            <table>
              <thead>
                <tr>
                  {['Mode', 'Prefill s', 'Decode s', 'Total s', 'Tokens/s', 'Output tokens', 'Peak GPU MB', 'Draft accepted', 'Tokens match'].map((heading) => (
                    <th key={heading}>{heading}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {output.rows.map((row) => (
                  <tr key={row.mode}>
                    <td>{MODE_NAMES[row.mode]}</td>
                    <td className="tabular-nums">{number(row.prefill_s, 3)}</td>
                    <td className="tabular-nums">{number(row.decode_s, 3)}</td>
                    <td className="tabular-nums">{number(row.total_s, 3)}</td>
                    <td className="tabular-nums">{number(row.tokens_per_s, 1)}</td>
                    <td className="tabular-nums">{row.output_tokens}</td>
                    <td className="tabular-nums">{number(row.peak_gpu_extra_mb, 1)}</td>
                    <td className="tabular-nums">{row.acceptance_rate == null ? 'n/a' : `${(row.acceptance_rate * 100).toFixed(1)}%`}</td>
                    <td>{row.tokens_match_reference ? 'Yes' : 'No'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="m-0 mt-3 text-sm text-neutral-400">
              Target on {output.device}. Draft on {output.draft_device}. One live run, so timings vary.
            </p>
          </div>
          <Bars title="Total time" unit="Seconds" rows={output.rows.map((row) => ({ label: MODE_NAMES[row.mode], value: row.total_s, reference: row.mode === 'uncached' }))} />
        </>
      )}
    </div>
  );
}

function AttentionLab() {
  const [length, setLength] = useState(512);
  const [tile, setTile] = useState(64);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [output, setOutput] = useState<AttentionOutput | null>(null);

  async function run() {
    setRunning(true);
    setError('');
    try {
      const { id } = await api.compareAttention({ length, tile });
      const job = await waitForJob<AttentionOutput>(id);
      if (job.state === 'done' && job.output) setOutput(job.output);
      else setError(job.error ?? 'The comparison did not finish.');
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="card grid items-end gap-4 p-5 sm:grid-cols-[1fr_1fr_auto]">
        <label className="flex flex-col gap-2 text-neutral-300">
          Sequence length
          <select className="field" value={length} onChange={(event) => setLength(Number(event.target.value))}>
            {[64, 256, 512, 1024, 2048, 4096].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-2 text-neutral-300">
          Tile size
          <select className="field" value={tile} onChange={(event) => setTile(Number(event.target.value))}>
            {[16, 32, 64, 128, 256].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <button type="button" className="btn btn-primary mb-1" disabled={running || tile > length} onClick={run}>
          {running ? 'Running...' : 'Run comparison'}
        </button>
      </div>
      {tile > length && <p className="m-0 text-sm text-neutral-400">The tile size cannot be larger than the sequence length.</p>}
      {error && <Notice message={error} />}
      {output && (
        <>
          <div className="card grid gap-4 p-5 sm:grid-cols-3">
            {[
              ['Max absolute error', output.max_abs_error.toExponential(2)],
              ['Mean absolute error', output.mean_abs_error.toExponential(2)],
              ['Tiles computed / skipped', `${output.tiles_computed} / ${output.tiles_skipped}`],
            ].map(([label, value]) => (
              <div key={label}>
                <p className="m-0 text-sm text-neutral-400">{label}</p>
                <p className="m-0 mt-1 text-xl tabular-nums">{value}</p>
              </div>
            ))}
          </div>
          <Bars
            title="Runtime"
            unit={`Milliseconds on ${output.device}, causal mask, length ${output.length}, tile ${output.tile}`}
            rows={[
              { label: 'Reference', value: output.reference_ms, reference: true },
              { label: 'Tiled online softmax', value: output.tiled_ms, reference: false },
            ]}
          />
          {output.reference_gpu_extra_mb != null && (
            <Bars
              title="Peak GPU memory above the inputs"
              unit="Megabytes"
              rows={[
                { label: 'Reference', value: output.reference_gpu_extra_mb, reference: true },
                { label: 'Tiled online softmax', value: output.tiled_gpu_extra_mb ?? 0, reference: false },
              ]}
            />
          )}
        </>
      )}
    </div>
  );
}

function Reports({ onOpen }: { onOpen: (id: string) => void }) {
  const [rows, setRows] = useState<ReportSummary[] | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.reports().then((body) => setRows(body.reports)).catch((failure: Error) => setError(failure.message));
  }, []);

  if (error) return <Notice message={error} />;
  if (!rows) return <p className="m-0 text-neutral-400">Loading...</p>;
  if (!rows.length) return <p className="m-0 text-neutral-400">No saved screenings yet.</p>;
  return (
    <div className="card overflow-x-auto p-5">
      <table>
        <thead>
          <tr>
            {['Created (UTC)', 'Outcome', 'Criteria', 'Mode', ''].map((heading) => (
              <th key={heading}>{heading}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td className="tabular-nums">{row.created_utc.replace('T', ' ').replace('+00:00', '')}</td>
              <td>{row.outcome}</td>
              <td className="tabular-nums">{row.criteria}</td>
              <td>{row.mode}</td>
              <td className="whitespace-nowrap">
                <button type="button" className="btn !px-3 !py-1 text-sm" onClick={() => onOpen(row.id)}>
                  Open
                </button>{' '}
                <a className="btn !px-3 !py-1 text-sm no-underline" href={`/api/reports/${row.id}/download.html`}>
                  Download
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Inspect({ status, onOpenReport, ...lab }: Omit<LabProps, 'ready'> & { status: Status | null; onOpenReport: (id: string) => void }) {
  const ready = status?.state === 'ready';
  const [tab, setTab] = useState<(typeof TABS)[number][0]>('decoding');
  return (
    <div className="flex w-full flex-col gap-8 px-8 pt-6 pb-36 lg:px-12">
      <div>
        <h1 className="glow-text m-0 text-4xl font-medium">Inspect</h1>
        <div className="mt-5 flex flex-wrap gap-2" role="tablist" aria-label="Inspect sections">
          {TABS.map(([id, label]) => (
            <button key={id} type="button" role="tab" id={`tab-${id}`} aria-selected={tab === id} aria-controls={`panel-${id}`} className={`btn text-sm ${tab === id ? 'btn-primary' : ''}`} onClick={() => setTab(id)}>
              {label}
            </button>
          ))}
        </div>
      </div>
      {status?.state === 'failed' && <Notice message={status.error ?? 'The model failed to load.'} />}
      {/* Panels stay mounted, so a finished comparison is still there after switching tabs. */}
      <Panel id="decoding" tab={tab} title="Decoding comparison">
        <DecodingLab ready={ready} {...lab} />
      </Panel>
      <Panel id="attention" tab={tab} title="Attention comparison">
        <AttentionLab />
      </Panel>
      <Panel id="reports" tab={tab} title="Saved reports">
        <Reports onOpen={onOpenReport} />
      </Panel>
      <Panel id="model" tab={tab} title="Model">
        {status && (
          <div className="card p-5">
            <table>
              <tbody>
                {[
                  ['Target model', status.target_model],
                  ['Fine-tuned adapter', status.adapter ?? 'None, untrained base model'],
                  ['Output format', status.output_order.replace('_', ' ')],
                  ['Draft model', status.draft_model],
                  ['Device', status.device],
                  ['Draft device', status.draft_device],
                ].map(([label, value]) => (
                  <tr key={label}>
                    <th>{label}</th>
                    <td>{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
