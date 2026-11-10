import { Download, FileJson, PencilLine } from 'lucide-react';
import type { CriterionResult, Overall } from './api';
import { Notice, Verdict } from './shared';

export type Screening = {
  overall: Overall;
  results: CriterionResult[];
  reportId: string;
  sentences: string[];
};

const OUTCOME_CARD: Record<string, string> = { 'Potentially eligible': 'card-matcha', 'Needs review': 'card-butter', 'Likely ineligible': 'card-pink' };

function countChips(counts: Record<string, number>) {
  return (
    [
      [counts.met, 'met', 'chip-met'],
      [counts.failed, 'failed', 'chip-failed'],
      [counts.unknown, 'unknown', 'chip-unknown'],
      [counts.not_applicable, 'not applicable', 'chip-na'],
      [counts.unverified, 'unverified', 'chip-invalid'],
    ] as Array<[number, string, string]>
  ).filter(([value]) => value);
}

function Group({ title, items, onOpen }: { title: string; items: CriterionResult[]; onOpen: (index: number) => void }) {
  if (!items.length) return null;
  return (
    <section className="card overflow-hidden">
      <h2 className="m-0 border-b-2 border-[var(--line)] bg-[var(--paper)] px-5 py-3 !text-lg">{title}</h2>
      {items.map((result) => (
        <button key={result.index} type="button" className="row-link flex items-center justify-between gap-5 px-5 py-3.5" onClick={() => onOpen(result.index)}>
          <span className="flex min-w-0 items-baseline gap-3">
            <span className="mono shrink-0 font-bold text-[var(--muted)]">#{result.index + 1}</span>
            <span className="min-w-0">{result.criterion}</span>
          </span>
          <span className="flex shrink-0 flex-col items-end gap-1">
            <Verdict label={result.label} />
            <span className="hint mono">{result.evidence_ids.length ? `cites ${result.evidence_ids.join(', ')}` : 'no citation'}</span>
          </span>
        </button>
      ))}
    </section>
  );
}

export function Results({ screening, onOpen, onEdit }: { screening: Screening; onOpen: (index: number) => void; onEdit: () => void }) {
  const { overall, results, reportId } = screening;
  return (
    <div className="flex flex-col gap-6">
      <p className="eyebrow">Step 4 of 4 · Results</p>
      <div className={`card ${OUTCOME_CARD[overall.outcome] ?? 'card-butter'} flex flex-col gap-4 p-6 md:flex-row md:items-end md:justify-between`}>
        <div className="flex flex-col gap-2">
          <p className="eyebrow">Preliminary overall outcome</p>
          <h1 className="m-0 !text-[2.75rem]">{overall.outcome}</h1>
          {overall.reason && <p className="m-0 text-[1.0625rem]">{overall.reason}</p>}
          <div className="mt-1 flex flex-wrap gap-2">
            {countChips(overall.counts).map(([value, label, chip]) => (
              <span key={label} className={`chip ${chip}`}>
                {value} {label}
              </span>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap gap-2 md:flex-col">
          <a className="btn btn-primary" href={`/api/reports/${reportId}/download.html`}>
            <Download size={16} aria-hidden="true" />
            Download report
          </a>
          <a className="btn" href={`/api/reports/${reportId}/download.json`}>
            <FileJson size={16} aria-hidden="true" />
            Download JSON
          </a>
          <button type="button" className="btn" onClick={onEdit}>
            <PencilLine size={16} aria-hidden="true" />
            Change criteria
          </button>
        </div>
      </div>
      <p className="hint">Select any criterion to see the exact sentences it relied on.</p>
      <Group title="Inclusion criteria" items={results.filter((item) => item.criterion_type === 'inclusion')} onOpen={onOpen} />
      <Group title="Exclusion criteria" items={results.filter((item) => item.criterion_type === 'exclusion')} onOpen={onOpen} />
      <div className="card card-butter p-4">
        <p className="eyebrow mb-1">How the overall outcome is decided</p>
        <p className="m-0 text-sm">{overall.rule}</p>
      </div>
    </div>
  );
}

const timingLabels: Record<string, string> = {
  mode: 'Decoding method',
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
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <button type="button" className="btn btn-small" onClick={onBack}>
          ← All results
        </button>
        <div className="flex items-center gap-2">
          <button type="button" className="btn btn-small" disabled={index === 0} onClick={() => onMove(index - 1)}>
            Previous
          </button>
          <span className="hint mono">
            {index + 1} / {total}
          </span>
          <button type="button" className="btn btn-small" disabled={index === total - 1} onClick={() => onMove(index + 1)}>
            Next
          </button>
        </div>
      </div>
      <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        <div className="flex flex-col gap-5">
          <div className="card flex flex-col gap-3 p-5">
            <p className="eyebrow">{result.criterion_type} criterion</p>
            <h1 className="m-0 !text-[1.625rem]">{result.criterion}</h1>
            <Verdict label={result.label} className="self-start !text-base" />
          </div>
          <div className="card card-matcha p-5">
            <p className="card-title">Why</p>
            <p className="m-0">{result.explanation}</p>
            {result.adjusted && <p className="hint mt-2">This label was corrected by the deterministic number check.</p>}
          </div>
          {missing.length > 0 && (
            <section className="flex flex-col gap-2">
              <p className="card-title !mb-0">Information the record does not provide</p>
              {missing.map((text) => (
                <Notice key={text} message={text} />
              ))}
            </section>
          )}
          <details className="card p-4">
            <summary>Technical details</summary>
            <div className="mt-3 flex flex-col gap-4">
              <table>
                <tbody>
                  {Object.entries(result.timing).map(([key, value]) => (
                    <tr key={key}>
                      <th>{timingLabels[key] ?? key}</th>
                      <td className="mono">{typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(3) : value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div>
                <p className="hint mb-1 !font-bold">Raw model output</p>
                <pre className="mono m-0 overflow-x-auto rounded-lg border-2 border-[var(--line)] bg-[var(--paper)] p-3 whitespace-pre-wrap">{result.raw_output}</pre>
              </div>
              {result.errors.length > 0 && <p className="m-0 text-sm">{result.errors.join(' ')}</p>}
            </div>
          </details>
        </div>
        <section className="flex flex-col gap-2">
          <p className="card-title !mb-0">Patient record</p>
          <p className="hint">{cited.size ? 'Highlighted sentences are the evidence the model cited.' : 'The model cited no sentence for this criterion.'}</p>
          <ol className="card m-0 flex list-none flex-col gap-1 p-3">
            {screening.sentences.map((sentence, id) => (
              <li key={id} className={`sentence flex gap-3 px-2 py-1.5 ${cited.has(id) ? 'sentence-lit' : cited.size ? 'sentence-dim' : ''}`}>
                <span className="mono w-6 shrink-0 text-right font-bold">{id}</span>
                <span>{sentence}</span>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </div>
  );
}
