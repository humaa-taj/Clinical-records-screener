import type { ReactNode } from 'react';
import type { Status } from './api';

const MET = ['included', 'not excluded'];
const FAILED = ['excluded', 'not included'];

export const NOTICE = 'Educational screening prototype. Not a clinical enrollment decision.';

export function verdictKind(label: string) {
  if (MET.includes(label)) return 'met';
  if (FAILED.includes(label)) return 'failed';
  if (label === 'not enough information') return 'unknown';
  if (label === 'not applicable') return 'na';
  return 'invalid';
}

export function verdictText(label: string) {
  const text = label === '__invalid__' ? 'unverified output' : label;
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function Verdict({ label, className = '' }: { label: string; className?: string }) {
  return <span className={`chip chip-${verdictKind(label)} ${className}`}>{verdictText(label)}</span>;
}

export function statusView(status: Status | null) {
  if (!status) return { kind: 'unknown', text: 'Connecting to server' };
  if (status.state === 'loading') return { kind: 'unknown', text: 'Model is loading' };
  if (status.state === 'failed') return { kind: 'failed', text: 'Model could not load' };
  return { kind: 'met', text: status.trained ? 'Fine-tuned model ready' : 'Base model ready (untrained)' };
}

export function StatusPill({ status }: { status: Status | null }) {
  const { kind, text } = statusView(status);
  return (
    <p className="m-0 inline-flex items-center gap-2 text-sm font-semibold" role="status">
      <span className={`dot dot-${kind}`} aria-hidden="true" />
      {text}
    </p>
  );
}

export function PageTitle({ eyebrow, title, lead }: { eyebrow?: string; title: string; lead?: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      {eyebrow && <p className="eyebrow">{eyebrow}</p>}
      <h1 className="m-0">{title}</h1>
      {lead && <p className="lead">{lead}</p>}
    </div>
  );
}

export function Notice({ message }: { message: string }) {
  return (
    <div className="card card-pink px-4 py-3 text-sm font-semibold" role="alert">
      {message}
    </div>
  );
}

export function ConfirmDialog({ title, message, confirm, onConfirm, onCancel }: { title: string; message: string; confirm: string; onConfirm: () => void; onCancel: () => void }) {
  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-[#2c3026]/40 px-4" role="dialog" aria-modal="true" aria-label={title}>
      <div className="card card-butter max-w-sm p-6">
        <h2 className="m-0">{title}</h2>
        <p className="m-0 mt-2 text-[0.9375rem]">{message}</p>
        <div className="mt-5 flex justify-end gap-3">
          <button type="button" className="btn" onClick={onCancel} autoFocus>
            Keep working
          </button>
          <button type="button" className="btn btn-pink" onClick={onConfirm}>
            {confirm}
          </button>
        </div>
      </div>
    </div>
  );
}

export function StepFooter({ onBack, nextLabel, onNext, nextDisabled }: { onBack?: () => void; nextLabel: string; onNext: () => void; nextDisabled?: boolean }) {
  return (
    <div className="mt-2 flex items-center justify-between gap-3 border-t-2 border-[var(--soft-line)] pt-5">
      {onBack ? (
        <button type="button" className="btn" onClick={onBack}>
          ← Back
        </button>
      ) : (
        <span />
      )}
      <button type="button" className="btn btn-primary" onClick={onNext} disabled={nextDisabled}>
        {nextLabel}
      </button>
    </div>
  );
}
