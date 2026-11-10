import { Combine, FileUp, Plus, Scissors, Sparkles, Trash2 } from 'lucide-react';
import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { api, waitForJob, type Status } from './api';
import { Notice, PageTitle, StepFooter } from './shared';

export type Draft = {
  note: string;
  /** The exact note the current sentences were split from. */
  splitNote: string;
  sentences: string[];
  inclusion: string;
  exclusion: string;
  mode: string;
};

const MAX_FILE_BYTES = 200_000;
const lines = (text: string) => text.split('\n').map((line) => line.trim()).filter(Boolean);

export function NoteStep({ draft, update, onNext }: { draft: Draft; update: (patch: Partial<Draft>) => void; onNext: () => void }) {
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function next() {
    // An unchanged note keeps the sentences, including any edits made on the next screen.
    if (draft.sentences.length && draft.note === draft.splitNote) {
      onNext();
      return;
    }
    setBusy(true);
    setError('');
    try {
      const { sentences } = await api.sentences(draft.note);
      update({ sentences, splitNote: draft.note });
      onNext();
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function loadExample() {
    try {
      const example = await api.example();
      update({ note: example.note, inclusion: example.inclusion.join('\n'), exclusion: example.exclusion.join('\n') });
      setError('');
    } catch (failure) {
      setError((failure as Error).message);
    }
  }

  async function onFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) {
      setError('This file is over 200 KB. Please choose a smaller plain text file.');
      return;
    }
    if (!/\.(txt|md)$/i.test(file.name)) {
      setError('Only .txt or .md files can be read. You can also paste the record directly.');
      return;
    }
    update({ note: await file.text() });
    setError('');
  }

  return (
    <div className="flex flex-col gap-6">
      <PageTitle eyebrow="Step 1 of 4" title="Add the patient record" lead="Paste a free-text clinical note. The model will only use what is written here." />
      <div className="grid gap-5 lg:grid-cols-[1fr_16rem]">
        <label className="field-label">
          Patient record
          <textarea
            className="field min-h-80"
            value={draft.note}
            onChange={(event) => update({ note: event.target.value })}
            placeholder="e.g. A 58-year-old woman with type 2 diabetes presents with..."
            aria-label="Patient note"
          />
        </label>
        <aside className="card card-matcha flex flex-col gap-3 self-start p-4">
          <p className="card-title !mb-0">Other ways in</p>
          <label className="btn w-full cursor-pointer">
            <FileUp size={16} aria-hidden="true" />
            Upload a .txt file
            <input type="file" accept=".txt,.md,text/plain" className="sr-only" onChange={onFile} />
          </label>
          <button type="button" className="btn w-full" onClick={loadExample}>
            <Sparkles size={16} aria-hidden="true" />
            Fill in a sample case
          </button>
          <p className="hint">The sample also fills in matching trial criteria for step 3.</p>
        </aside>
      </div>
      {error && <Notice message={error} />}
      <StepFooter nextLabel={busy ? 'Splitting into sentences…' : 'Next: check sentences →'} onNext={next} nextDisabled={busy} />
    </div>
  );
}

export function SentenceStep({ draft, update, onBack, onNext }: { draft: Draft; update: (patch: Partial<Draft>) => void; onBack: () => void; onNext: () => void }) {
  const [error, setError] = useState('');
  const inputs = useRef<Array<HTMLInputElement | null>>([]);
  const { sentences } = draft;
  const change = (index: number, text: string) => {
    update({ sentences: sentences.map((value, i) => (i === index ? text : value)) });
    setError('');
  };
  const remove = (index: number) => update({ sentences: sentences.filter((_, i) => i !== index) });
  const add = () => update({ sentences: [...sentences, ''] });

  function mergeWithNext(index: number) {
    const merged = `${sentences[index].trim()} ${sentences[index + 1].trim()}`.trim();
    update({ sentences: [...sentences.slice(0, index), merged, ...sentences.slice(index + 2)] });
    setError('');
  }

  function split(index: number) {
    const text = sentences[index];
    const at = inputs.current[index]?.selectionStart ?? 0;
    const [left, right] = [text.slice(0, at).trim(), text.slice(at).trim()];
    if (!left || !right) {
      setError(`Place the cursor inside sentence ${index} where the break should go, then press the scissors.`);
      return;
    }
    update({ sentences: [...sentences.slice(0, index), left, right, ...sentences.slice(index + 1)] });
    setError('');
  }

  function next() {
    if (!draft.sentences.length || draft.sentences.some((sentence) => !sentence.trim())) {
      setError('One or more sentences are empty. Type some text or delete them.');
      return;
    }
    setError('');
    onNext();
  }

  return (
    <div className="flex flex-col gap-6">
      <PageTitle
        eyebrow="Step 2 of 4"
        title="Check the numbered sentences"
        lead="Every verdict points back to these numbers as evidence. Fix any sentence that was broken in the wrong place."
      />
      <div className="card card-butter flex flex-wrap gap-x-6 gap-y-1 px-4 py-3 text-sm">
        <span className="inline-flex items-center gap-2"><Scissors size={14} aria-hidden="true" /> split at cursor</span>
        <span className="inline-flex items-center gap-2"><Combine size={14} aria-hidden="true" /> join with next</span>
        <span className="inline-flex items-center gap-2"><Trash2 size={14} aria-hidden="true" /> delete</span>
      </div>
      <div className="card flex flex-col gap-2 p-4">
        <ol className="m-0 flex list-none flex-col gap-2 p-0">
          {sentences.map((sentence, index) => (
            <li key={index} className="flex items-center gap-2">
              <span className="mono w-8 shrink-0 text-right font-bold text-[var(--muted)]">{index}</span>
              <input ref={(element) => { inputs.current[index] = element; }} className="field !py-2" value={sentence} onChange={(event) => change(index, event.target.value)} aria-label={`Sentence ${index}`} />
              <button type="button" className="icon-btn" onClick={() => split(index)} aria-label={`Split sentence ${index} at the cursor`} title="Split at the cursor">
                <Scissors size={16} aria-hidden="true" />
              </button>
              <button type="button" className="icon-btn" disabled={index === sentences.length - 1} onClick={() => mergeWithNext(index)} aria-label={`Merge sentence ${index} with the next one`} title="Join with the next sentence">
                <Combine size={16} aria-hidden="true" />
              </button>
              <button type="button" className="icon-btn" onClick={() => remove(index)} aria-label={`Remove sentence ${index}`} title="Delete">
                <Trash2 size={16} aria-hidden="true" />
              </button>
            </li>
          ))}
        </ol>
        <button type="button" className="btn btn-small mt-2 self-start" onClick={add}>
          <Plus size={14} aria-hidden="true" />
          Add a sentence
        </button>
      </div>
      {error && <Notice message={error} />}
      <StepFooter onBack={onBack} nextLabel="Next: trial criteria →" onNext={next} />
    </div>
  );
}

const MODES = [
  ['cached', 'KV cache — fastest'],
  ['speculative', 'Speculative decoding'],
  ['uncached', 'No cache — reference'],
];

export function CriteriaStep({ draft, update, status, onBack, onRun }: { draft: Draft; update: (patch: Partial<Draft>) => void; status: Status | null; onBack: () => void; onRun: () => Promise<void> }) {
  const [error, setError] = useState('');
  const [trial, setTrial] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const ready = status?.state === 'ready';

  async function importTrial() {
    setError('');
    setMessage('');
    setBusy(true);
    try {
      const found = await api.trial(trial);
      update({ inclusion: found.inclusion.join('\n'), exclusion: found.exclusion.join('\n') });
      setMessage(found.title ? `Imported “${found.title}”` : `Imported ${found.id}`);
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function run() {
    if (!lines(draft.inclusion).length && !lines(draft.exclusion).length) {
      setError('Add at least one inclusion or exclusion criterion before running.');
      return;
    }
    setError('');
    setBusy(true);
    try {
      await onRun();
    } catch (failure) {
      setError((failure as Error).message);
      setBusy(false);
    }
  }

  const statusText = status?.state === 'failed' ? status.error : status?.state === 'loading' || !status ? 'The model is still loading. The run button unlocks when it is ready.' : '';
  return (
    <div className="flex flex-col gap-6">
      <PageTitle eyebrow="Step 3 of 4" title="Enter the trial criteria" lead="Write one criterion per line. Each line is judged on its own against the patient record." />
      <div className="grid gap-5 lg:grid-cols-[1fr_18rem]">
        <div className="grid gap-4 md:grid-cols-2">
          <label className="field-label">
            <span className="inline-flex items-center gap-2"><span className="dot dot-met" aria-hidden="true" />Inclusion — patient must meet</span>
            <textarea className="field min-h-64" value={draft.inclusion} onChange={(event) => update({ inclusion: event.target.value })} />
          </label>
          <label className="field-label">
            <span className="inline-flex items-center gap-2"><span className="dot dot-failed" aria-hidden="true" />Exclusion — patient must not have</span>
            <textarea className="field min-h-64" value={draft.exclusion} onChange={(event) => update({ exclusion: event.target.value })} />
          </label>
        </div>
        <aside className="flex flex-col gap-4 self-start">
          <div className="card card-pink flex flex-col gap-2 p-4">
            <label htmlFor="trial-id" className="card-title !mb-0">Import a real trial</label>
            <p className="hint">Pulls criteria from ClinicalTrials.gov by NCT number.</p>
            <div className="flex gap-2">
              <input id="trial-id" className="field" placeholder="NCT01234567" value={trial} onChange={(event) => setTrial(event.target.value)} />
              <button type="button" className="btn shrink-0" onClick={importTrial} disabled={busy || !trial.trim()}>
                Import
              </button>
            </div>
            {message && <p className="hint !font-semibold !text-[var(--ink)]">{message}</p>}
          </div>
          <div className="card card-matcha flex flex-col gap-2 p-4">
            <label className="field-label">
              <span className="card-title !mb-0">Decoding method</span>
              <select className="field" value={draft.mode} onChange={(event) => update({ mode: event.target.value })}>
                {MODES.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
            </label>
            {draft.mode === 'speculative' && <p className="hint">Produces exactly the same answer as the KV cache. In our benchmarks it was slower on this hardware.</p>}
            {draft.mode === 'uncached' && <p className="hint">Recomputes everything at each token. Correct but slow; useful as a reference.</p>}
          </div>
        </aside>
      </div>
      {statusText && <p className="hint">{statusText}</p>}
      {error && <Notice message={error} />}
      <StepFooter onBack={onBack} nextLabel={busy ? 'Starting…' : 'Run screening →'} onNext={run} nextDisabled={busy || !ready} />
    </div>
  );
}

export function Running({ jobId, onDone, onFailed, onCancelled }: { jobId: string; onDone: (job: Awaited<ReturnType<typeof waitForJob>>) => void; onFailed: (message: string) => void; onCancelled: () => void }) {
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [cancelling, setCancelling] = useState(false);
  const finished = useRef(false);

  useEffect(() => {
    let active = true;
    waitForJob(jobId, (job) => active && setProgress({ done: job.done, total: job.total }))
      .then((job) => {
        finished.current = true;
        if (!active) return;
        if (job.state === 'done') onDone(job);
        else if (job.state === 'cancelled') onCancelled();
        else onFailed(job.error ?? 'The screening did not finish. Please try again.');
      })
      .catch((failure: Error) => {
        finished.current = true;
        if (active) onFailed(failure.message);
      });
    return () => {
      active = false;
      if (!finished.current) api.cancel(jobId).catch(() => undefined);
    };
  }, [jobId]);

  const current = Math.min(progress.done + 1, progress.total);
  const percent = progress.total ? Math.round((progress.done / progress.total) * 100) : 0;
  return (
    <div className="mx-auto flex w-full max-w-xl flex-col gap-6 pt-8" role="status" aria-live="polite">
      <PageTitle eyebrow="Screening in progress" title={progress.total ? `Reading criterion ${current} of ${progress.total}` : 'Getting started'} lead="On a CPU each criterion can take several seconds. You can cancel at any time." />
      <div className="card card-butter flex flex-col gap-3 p-5">
        <div className="progress" aria-hidden="true">
          <span style={{ width: `${percent}%` }} />
        </div>
        <p className="hint mono">
          {progress.done} / {progress.total || '?'} done
        </p>
      </div>
      <button
        type="button"
        className="btn btn-pink self-start"
        disabled={cancelling}
        onClick={() => {
          setCancelling(true);
          api.cancel(jobId).catch(() => setCancelling(false));
        }}
      >
        {cancelling ? 'Stopping…' : 'Cancel screening'}
      </button>
    </div>
  );
}
