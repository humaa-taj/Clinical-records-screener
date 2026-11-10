import type { CriterionResult, Overall } from './api';
import { Notice, Verdict } from './shared';

export type Screening = {
  overall: Overall;
  results: CriterionResult[];
  reportId: string;
  sentences: string[];
};

function countLine(counts: Record<string, number>) {
  return [
    [counts.met, 'met'],
    [counts.failed, 'failed'],
    [counts.unknown, 'unknown'],
    [counts.not_applicable, 'not applicable'],
    [counts.unverified, 'unverified'],
  ]
    .filter(([value]) => value)
    .map(([value, label]) => `${value} ${label}`)
    .join(' · ');
}

export function Results({ screening, onOpen, onEdit }: { screening: Screening; onOpen: (index: number) => void; onEdit: () => void }) {
  const { overall, results, reportId } = screening;
  return (
    <div className="flex w-full max-w-3xl flex-col items-center gap-7">
      <div className="text-center">
        <p className="m-0 text-sm tracking-wide text-neutral-400 uppercase">Overall · preliminary</p>
        <h1 className="glow-text m-0 mt-2 text-6xl font-medium">{overall.outcome}</h1>
        <p className="m-0 mt-3 text-lg text-neutral-300 tabular-nums">{countLine(overall.counts)}</p>
        {overall.reason && <p className="m-0 mt-1 text-neutral-300">{overall.reason}</p>}
      </div>
      <div className="flex flex-wrap justify-center gap-3">
        <button type="button" className="btn" onClick={onEdit}>
          Edit criteria
        </button>
        <a className="btn no-underline" href={`/api/reports/${reportId}/download.html`}>
          Download report
        </a>
        <a className="btn no-underline" href={`/api/reports/${reportId}/download.json`}>
          Download JSON
        </a>
      </div>
      <div className="card w-full overflow-hidden">
        {results.map((result) => (
          <button key={result.index} type="button" className="row-link flex items-center justify-between gap-5 px-5 py-3.5" onClick={() => onOpen(result.index)}>
            <span className="min-w-0">
              <span className="block text-xs text-neutral-400 uppercase">{result.criterion_type}</span>
              <span className="block">{result.criterion}</span>
            </span>
            <span className="flex shrink-0 flex-col items-end gap-0.5">
              <Verdict label={result.label} />
              <span className="text-xs text-neutral-400 tabular-nums">
                {result.evidence_ids.length ? `${result.evidence_ids.length} cited` : 'none cited'}
              </span>
            </span>
          </button>
        ))}
      </div>
      <p className="m-0 max-w-2xl text-center text-sm leading-relaxed text-neutral-400">{overall.rule}</p>
    </div>
  );
}

function Detail({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="m-0 mb-2 text-base font-medium text-neutral-300">{title}</h2>
      {children}
    </section>
  );
}

const timingLabels: Record<string, string> = {
  mode: 'Decoding mode',
  input_tokens: 'Input tokens',
  output_tokens: 'Output tokens',
  prefill_s: 'Prefill (s)',
  decode_s: 'Decode (s)',
  total_s: 'Total (s)',
  tokens_per_s: 'Tokens per second',
  acceptance_rate: 'Draft acceptance',
  rounds: 'Speculative rounds',
};

export function CriterionDetail({ screening, index, onBack, onMove }: { screening: Screening; index: number; onBack: () => void; onMove: (index: number) => void }) {
  const result = screening.results[index];
  const cited = new Set(result.evidence_ids);
  const total = screening.results.length;
  // When the explanation already says it, repeating it in a second box adds nothing.
  const missing = result.missing_information.filter((text) => !result.explanation.includes(text) && !(result.label === 'not enough information' && result.evidence_ids.length));
  return (
    <div className="flex w-full max-w-3xl flex-col gap-6">
      <div>
        <p className="m-0 text-sm text-neutral-400 uppercase">
          {result.criterion_type} · criterion {index + 1} of {total}
        </p>
        <h1 className="m-0 mt-1 text-2xl leading-snug font-medium">{result.criterion}</h1>
        <Verdict label={result.label} className="glow-text mt-3 text-3xl" />
      </div>
      <div>
        <p className="m-0 leading-relaxed text-neutral-200">{result.explanation}</p>
        {result.adjusted && <p className="m-0 mt-1 text-sm text-neutral-400">Adjusted by number check.</p>}
      </div>
      {missing.length > 0 && (
        <Detail title="Missing information">
          {missing.map((text) => (
            <Notice key={text} message={text} />
          ))}
        </Detail>
      )}
      <Detail title="Patient note">
        <ol className="card m-0 flex list-none flex-col gap-1 p-4">
          {screening.sentences.map((sentence, id) => (
            <li key={id} className={`sentence flex gap-3 px-2 py-1.5 ${cited.has(id) ? 'sentence-lit' : cited.size ? 'sentence-dim' : ''}`}>
              <span className="mono w-6 shrink-0 text-right text-neutral-400">{id}</span>
              <span>{sentence}</span>
            </li>
          ))}
        </ol>
      </Detail>
      <details className="card px-5 py-3">
        <summary className="cursor-pointer text-neutral-300">Technical details</summary>
        <div className="mt-3 flex flex-col gap-4">
          <table>
            <tbody>
              {Object.entries(result.timing).map(([key, value]) => (
                <tr key={key}>
                  <th>{timingLabels[key] ?? key}</th>
                  <td className="tabular-nums">{typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(3) : value}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div>
            <p className="m-0 mb-1 text-sm text-neutral-400">Raw model output</p>
            <pre className="mono m-0 overflow-x-auto text-sm whitespace-pre-wrap">{result.raw_output}</pre>
          </div>
          {result.errors.length > 0 && <p className="m-0 text-sm text-neutral-300">{result.errors.join(' ')}</p>}
        </div>
      </details>
      <div className="flex items-center justify-between">
        <button type="button" className="btn" onClick={onBack}>
          Back to results
        </button>
        <div className="flex gap-2">
          <button type="button" className="btn" disabled={index === 0} onClick={() => onMove(index - 1)}>
            Previous
          </button>
          <button type="button" className="btn" disabled={index === total - 1} onClick={() => onMove(index + 1)}>
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
