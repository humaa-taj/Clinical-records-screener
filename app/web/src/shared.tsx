import type { Status } from './api';

const MET = ['included', 'not excluded'];
const FAILED = ['excluded', 'not included'];

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
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <span className={`dot dot-${verdictKind(label)}`} aria-hidden="true" />
      {verdictText(label)}
    </span>
  );
}

function statusView(status: Status | null) {
  if (!status) return { kind: 'unknown', text: 'Connecting' };
  if (status.state === 'loading') return { kind: 'unknown', text: 'Model loading' };
  if (status.state === 'failed') return { kind: 'failed', text: 'Model failed to load' };
  return { kind: 'met', text: status.trained ? 'Model ready' : 'Untrained model ready' };
}

export function PageHeader({ status }: { status: Status | null }) {
  const { kind, text } = statusView(status);
  return (
    <header className="flex items-center justify-between gap-4 px-8 pt-7">
      <p className="display m-0 text-xl font-medium tracking-wide">CRS</p>
      <p className="m-0 inline-flex items-center gap-2 text-sm text-neutral-300" role="status">
        <span className={`dot dot-${kind}`} aria-hidden="true" />
        {text}
      </p>
    </header>
  );
}

export function Notice({ message }: { message: string }) {
  return (
    <div className="card px-5 py-4 text-center text-neutral-100" role="alert">
      {message}
    </div>
  );
}

export function ConfirmDialog({ title, message, confirm, onConfirm, onCancel }: { title: string; message: string; confirm: string; onConfirm: () => void; onCancel: () => void }) {
  return (
    <div className="fixed inset-0 z-30 flex items-center justify-center bg-black/70 px-4" role="dialog" aria-modal="true" aria-label={title}>
      <div className="card max-w-sm bg-black p-6 text-center">
        <h2 className="m-0 text-xl font-medium">{title}</h2>
        <p className="m-0 mt-2 text-neutral-300">{message}</p>
        <div className="mt-5 flex justify-center gap-3">
          <button type="button" className="btn" onClick={onCancel} autoFocus>
            Stay
          </button>
          <button type="button" className="btn btn-primary" onClick={onConfirm}>
            {confirm}
          </button>
        </div>
      </div>
    </div>
  );
}

export function StepFooter({ onBack, nextLabel, onNext, nextDisabled }: { onBack?: () => void; nextLabel: string; onNext: () => void; nextDisabled?: boolean }) {
  return (
    <div className="mt-2 flex items-center justify-between">
      {onBack ? (
        <button type="button" className="btn" onClick={onBack}>
          Back
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
