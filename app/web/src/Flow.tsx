import { Combine, Plus, Scissors, Trash2 } from 'lucide-react';
import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { ThinkingOrb } from 'thinking-orbs';
import { api, waitForJob, type Status } from './api';
import { Notice, StepFooter } from './shared';

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
      setError('That file is too large. Use a plain text file under 200 KB.');
      return;
    }
    if (!/\.(txt|md)$/i.test(file.name)) {
      setError('Use a plain text file ending in .txt or .md, or paste the note instead.');
      return;
    }
    update({ note: await file.text() });
    setError('');
  }

  return (
    <div className="flex w-full max-w-2xl flex-col gap-5">
      <h1 className="glow-text m-0 text-3xl font-medium">Patient note</h1>
      <textarea
        className="field min-h-72 leading-relaxed"
        value={draft.note}
        onChange={(event) => update({ note: event.target.value })}
        placeholder="Paste the clinical note"
        aria-label="Patient note"
      />
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <label className="btn cursor-pointer">
          Upload text file
          <input type="file" accept=".txt,.md,text/plain" className="sr-only" onChange={onFile} />
        </label>
        <button type="button" className="btn" onClick={loadExample}>
          Use example
        </button>
      </div>
      {error && <Notice message={error} />}
      <StepFooter nextLabel={busy ? 'Reading...' : 'Continue'} onNext={next} nextDisabled={busy} />
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
      setError(`Click inside sentence ${index} where it should split, then press split.`);
      return;
    }
    update({ sentences: [...sentences.slice(0, index), left, right, ...sentences.slice(index + 1)] });
    setError('');
  }

  function next() {
    if (!draft.sentences.length || draft.sentences.some((sentence) => !sentence.trim())) {
      setError('Every sentence needs text. Remove any empty sentences.');
      return;
    }
    setError('');
    onNext();
  }

  return (
    <div className="flex w-full max-w-3xl flex-col gap-5">
      <div>
        <h1 className="glow-text m-0 text-3xl font-medium">Review sentences</h1>
        <p className="m-0 mt-2 text-neutral-400">Each verdict will cite these numbers. Fix any sentence that was cut in the wrong place.</p>
      </div>
      <div className="card flex flex-col gap-2 p-4">
        <ol className="m-0 flex list-none flex-col gap-2 p-0">
          {sentences.map((sentence, index) => (
            <li key={index} className="flex items-center gap-2">
              <span className="mono w-7 shrink-0 text-right text-neutral-400">{index}</span>
              <input ref={(element) => { inputs.current[index] = element; }} className="field !py-2" value={sentence} onChange={(event) => change(index, event.target.value)} aria-label={`Sentence ${index}`} />
              <button type="button" className="icon-btn" onClick={() => split(index)} aria-label={`Split sentence ${index} at the cursor`} title="Split at the cursor">
                <Scissors size={16} aria-hidden="true" />
              </button>
              <button type="button" className="icon-btn" disabled={index === sentences.length - 1} onClick={() => mergeWithNext(index)} aria-label={`Merge sentence ${index} with the next one`} title="Merge with the next sentence">
                <Combine size={16} aria-hidden="true" />
              </button>
              <button type="button" className="icon-btn" onClick={() => remove(index)} aria-label={`Remove sentence ${index}`} title="Remove">
                <Trash2 size={16} aria-hidden="true" />
              </button>
            </li>
          ))}
        </ol>
        <button type="button" className="btn mt-1 inline-flex items-center gap-2 self-start text-sm" onClick={add}>
          <Plus size={16} aria-hidden="true" />
          Add sentence
        </button>
      </div>
      {error && <Notice message={error} />}
      <StepFooter onBack={onBack} nextLabel="Continue" onNext={next} />
    </div>
  );
}

const MODES = [
  ['cached', 'KV cache (fastest)'],
  ['speculative', 'Speculative'],
  ['uncached', 'No cache (reference)'],
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
      setMessage(found.title ? `Loaded: ${found.title}` : `Loaded ${found.id}`);
    } catch (failure) {
      setError((failure as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function run() {
    if (!lines(draft.inclusion).length && !lines(draft.exclusion).length) {
      setError('Enter at least one inclusion or exclusion criterion.');
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

  const statusText = status?.state === 'failed' ? status.error : status?.state === 'loading' || !status ? 'The model is still loading.' : '';
  return (
    <div className="flex w-full max-w-3xl flex-col gap-5">
      <h1 className="glow-text m-0 text-3xl font-medium">Trial criteria</h1>
      <div className="grid gap-4 md:grid-cols-2">
        <label className="flex flex-col gap-2 text-neutral-300">
          Inclusion, one per line
          <textarea className="field min-h-52" value={draft.inclusion} onChange={(event) => update({ inclusion: event.target.value })} />
        </label>
        <label className="flex flex-col gap-2 text-neutral-300">
          Exclusion, one per line
          <textarea className="field min-h-52" value={draft.exclusion} onChange={(event) => update({ exclusion: event.target.value })} />
        </label>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <div className="flex flex-col gap-2 text-neutral-300">
          <label htmlFor="trial-id">Import from ClinicalTrials.gov</label>
          <div className="flex gap-2">
            <input id="trial-id" className="field" placeholder="NCT01234567" value={trial} onChange={(event) => setTrial(event.target.value)} />
            <button type="button" className="btn shrink-0" onClick={importTrial} disabled={busy || !trial.trim()}>
              Load
            </button>
          </div>
        </div>
        <label className="flex flex-col gap-2 text-neutral-300">
          Decoding
          <select className="field" value={draft.mode} onChange={(event) => update({ mode: event.target.value })}>
            {MODES.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
          {draft.mode === 'speculative' && <span className="text-sm text-neutral-400">Gives the same output as the KV cache. It was slower in our benchmarks on this hardware.</span>}
        </label>
      </div>
      {message &&<p className="m-0 text-sm text-neutral-300">{message}</p>}
      {statusText && <p className="m-0 text-sm text-neutral-400">{statusText}</p>}
      {error && <Notice message={error} />}
      <StepFooter onBack={onBack} nextLabel={busy ? 'Starting...' : 'Run screening'} onNext={run} nextDisabled={busy || !ready} />
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
        else onFailed(job.error ?? 'The run failed. Try again.');
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
  return (
    <div className="flex flex-col items-center gap-5" role="status" aria-live="polite">
      <div className="card flex items-center gap-4 !rounded-full py-3 pr-8 pl-4">
        <ThinkingOrb state="working" size={64} theme="dark" aria-hidden="true" />
        <span className="text-2xl text-neutral-300">{progress.total ? `Checking criterion ${current} of ${progress.total}` : 'Starting'}</span>
      </div>
      <button
        type="button"
        className="btn"
        disabled={cancelling}
        onClick={() => {
          setCancelling(true);
          api.cancel(jobId).catch(() => setCancelling(false));
        }}
      >
        {cancelling ? 'Cancelling...' : 'Cancel'}
      </button>
    </div>
  );
}
