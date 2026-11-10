import { FlaskConical, House, TriangleAlert } from 'lucide-react';
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { api, type Report, type Status } from './api';
import { CriteriaStep, NoteStep, Running, SentenceStep, type Draft } from './Flow';
import Inspect from './Inspect';
import { CriterionDetail, Results, type Screening } from './Results';
import { ConfirmDialog, NOTICE, Notice, PageTitle, StatusPill } from './shared';

const EMPTY: Draft = { note: '', splitNote: '', sentences: [], inclusion: '', exclusion: '', mode: 'cached' };
const STORAGE_KEY = 'crs-session';
const readRoute = () => window.location.hash.replace(/^#\/?/, '');
const go = (route: string) => {
  window.location.hash = `/${route}`;
};
const replace = (route: string) => window.location.replace(`#/${route}`);
const splitLines = (text: string) => text.split('\n').map((line) => line.trim()).filter(Boolean);

type Saved = { draft: Draft; jobId: string; reportId: string };

function loadSaved(): Saved {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) ?? 'null');
    if (saved) return { draft: { ...EMPTY, ...saved.draft }, jobId: saved.jobId ?? '', reportId: saved.reportId ?? '' };
  } catch {
    // Storage can be blocked or hold something unreadable. The app then starts empty.
  }
  return { draft: EMPTY, jobId: '', reportId: '' };
}

function store(saved: Saved) {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
  } catch {
    // Without storage a refresh loses the work, and everything else still functions.
  }
}

const toScreening = (report: Report): Screening => ({ overall: report.overall, results: report.results, reportId: report.id, sentences: report.patient_sentences.map((item) => item.text) });

export default function App() {
  const [saved] = useState(loadSaved);
  const [route, setRoute] = useState(readRoute);
  const [status, setStatus] = useState<Status | null>(null);
  const [draft, setDraft] = useState<Draft>(saved.draft);
  const [jobId, setJobId] = useState(saved.jobId);
  const [screening, setScreening] = useState<Screening | null>(null);
  const [restoring, setRestoring] = useState(Boolean(saved.reportId));
  const [failure, setFailure] = useState('');
  const [confirmHome, setConfirmHome] = useState(false);
  const mainRef = useRef<HTMLElement>(null);

  const update = useCallback((patch: Partial<Draft>) => setDraft((current) => ({ ...current, ...patch })), []);

  useEffect(() => {
    const onChange = () => setRoute(readRoute());
    window.addEventListener('hashchange', onChange);
    return () => window.removeEventListener('hashchange', onChange);
  }, []);

  useEffect(() => {
    let active = true;
    let timer = 0;
    const poll = async () => {
      try {
        const next = await api.status();
        if (!active) return;
        setStatus(next);
        if (next.state === 'loading') timer = window.setTimeout(poll, 1500);
      } catch {
        if (active) timer = window.setTimeout(poll, 3000);
      }
    };
    poll();
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    mainRef.current?.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }, [route]);

  useEffect(() => {
    if (!saved.reportId) return;
    api.report(saved.reportId)
      .then((report) => setScreening(toScreening(report)))
      .catch(() => undefined)
      .finally(() => setRestoring(false));
  }, [saved.reportId]);

  useEffect(() => {
    if (!restoring) store({ draft, jobId, reportId: screening?.reportId ?? '' });
  }, [restoring, draft, jobId, screening?.reportId]);

  const stepsUnlocked = draft.sentences.length > 0 && draft.note === draft.splitNote;
  const detailIndex = route.startsWith('detail/') ? Number(route.slice(7)) : -1;
  const inspectIndex = route.startsWith('inspect/') ? Number(route.slice(8)) : -1;
  const onInspect = route === 'inspect' || route.startsWith('inspect/');
  const inResultsArea = route === 'results' || route.startsWith('detail/') || route.startsWith('inspect/');

  // A refreshed or hand-typed address may point at a screen whose data no longer exists.
  useEffect(() => {
    // While a navigation is in flight the state already describes the next screen, so wait for it to land.
    if (restoring || readRoute() !== route) return;
    if ((route === 'sentences' || route === 'criteria') && !stepsUnlocked) replace('note');
    if (route === 'running' && !jobId) replace(stepsUnlocked ? 'criteria' : 'note');
    if ((route === 'results' || route.startsWith('detail/')) && !screening) replace('note');
    if (route.startsWith('detail/') && screening && !(Number.isInteger(detailIndex) && screening.results[detailIndex])) replace('results');
    if (route.startsWith('inspect/') && !screening?.results[inspectIndex]) replace('inspect');
  }, [restoring, route, stepsUnlocked, jobId, screening, detailIndex, inspectIndex]);

  const hasWork = Boolean(draft.note || draft.sentences.length || draft.inclusion || draft.exclusion || screening);

  function resetAndGoHome() {
    setDraft(EMPTY);
    setScreening(null);
    setJobId('');
    setFailure('');
    setConfirmHome(false);
    go('');
  }

  const requestHome = () => (hasWork ? setConfirmHome(true) : resetAndGoHome());

  async function startRun() {
    const { id } = await api.screen({
      sentences: draft.sentences,
      inclusion: splitLines(draft.inclusion),
      exclusion: splitLines(draft.exclusion),
      mode: draft.mode,
    });
    setFailure('');
    setJobId(id);
    go('running');
  }

  async function openReport(id: string) {
    try {
      const report = await api.report(id);
      const opened = toScreening(report);
      const note = opened.sentences.join(' ');
      const criteria = (kind: string) => report.results.filter((item) => item.criterion_type === kind).map((item) => item.criterion).join('\n');
      setScreening(opened);
      update({ note, splitNote: note, sentences: opened.sentences, inclusion: criteria('inclusion'), exclusion: criteria('exclusion') });
      go('results');
    } catch (error) {
      setFailure((error as Error).message);
    }
  }

  let body = null;
  if (route === 'note') body = <NoteStep draft={draft} update={update} onNext={() => go('sentences')} />;
  else if (route === 'sentences' && stepsUnlocked) body = <SentenceStep draft={draft} update={update} onBack={() => go('note')} onNext={() => go('criteria')} />;
  else if (route === 'criteria' && stepsUnlocked) body = <CriteriaStep draft={draft} update={update} status={status} onBack={() => go('sentences')} onRun={startRun} />;
  else if (route === 'running' && jobId)
    body = (
      <Running
        jobId={jobId}
        onDone={(job) => {
          setScreening({ overall: job.overall!, results: job.results, reportId: job.report_id!, sentences: draft.sentences });
          replace('results');
        }}
        onFailed={(message) => {
          setFailure(message);
          replace('failed');
        }}
        onCancelled={() => replace('criteria')}
      />
    );
  else if (route === 'failed')
    body = (
      <div className="flex max-w-xl flex-col gap-5">
        <PageTitle eyebrow="Screening stopped" title="Something went wrong" />
        <Notice message={failure || 'The screening did not finish. Please try again.'} />
        <button type="button" className="btn self-start" onClick={() => replace('criteria')}>
          ← Back to criteria
        </button>
      </div>
    );
  else if (route === 'results' && screening) body = <Results screening={screening} onOpen={(index) => go(`detail/${index}`)} onEdit={() => go('criteria')} />;
  else if (route.startsWith('detail/') && screening?.results[detailIndex])
    body = <CriterionDetail screening={screening} index={detailIndex} onBack={() => go('results')} onMove={(index) => replace(`detail/${index}`)} />;

  const running = route === 'running';
  const inspected = screening?.results[inspectIndex];

  type NavEntry = { id: string; label: string; num?: string; icon?: ReactNode; onSelect: () => void; active: boolean; disabled?: boolean };
  const steps: NavEntry[] = [
    { id: 'note', num: '1', label: 'Patient record', onSelect: () => go('note'), active: route === 'note', disabled: running },
    { id: 'sentences', num: '2', label: 'Sentences', onSelect: () => go('sentences'), active: route === 'sentences', disabled: running || !stepsUnlocked },
    { id: 'criteria', num: '3', label: 'Trial criteria', onSelect: () => go('criteria'), active: route === 'criteria' || running || route === 'failed', disabled: running || !stepsUnlocked },
    { id: 'results', num: '4', label: 'Results', onSelect: () => go('results'), active: route === 'results' || route.startsWith('detail/'), disabled: running || !screening },
  ];
  const tools: NavEntry[] = [
    {
      id: 'inspect',
      icon: <FlaskConical size={16} aria-hidden="true" />,
      label: inResultsArea ? 'Model lab (this result)' : 'Model lab',
      onSelect: () => go(inResultsArea ? `inspect/${Math.max(detailIndex, 0)}` : 'inspect'),
      active: onInspect,
      disabled: running,
    },
  ];
  const navButton = ({ id, label, num, icon, onSelect, active, disabled }: NavEntry) => (
    <button key={id} type="button" className="nav-item" aria-current={active ? 'page' : undefined} disabled={disabled} onClick={onSelect}>
      {num ? <span className="nav-num">{num}</span> : icon}
      {label}
    </button>
  );

  let content: ReactNode = null;
  if (route === '')
    content = (
      <div className="flex flex-col gap-8">
        <div className="card card-pink flex flex-col gap-4 p-6 sm:p-8">
          <p className="eyebrow">Clinical trial eligibility, one criterion at a time</p>
          <h1 className="m-0 !text-[2.75rem]">Screen a patient record against trial criteria</h1>
          <p className="lead max-w-2xl !text-[var(--ink)]">
            A fine-tuned Qwen2.5-1.5B model reads the record, judges every inclusion and exclusion criterion, and cites the exact sentences behind each
            verdict.
          </p>
          <div className="flex flex-wrap gap-3">
            <button type="button" className="btn btn-primary" onClick={() => go('note')}>
              Start a screening →
            </button>
            <button type="button" className="btn" onClick={() => go('inspect')}>
              <FlaskConical size={16} aria-hidden="true" />
              Open the model lab
            </button>
          </div>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {[
            ['1', 'Add the record', 'Paste, upload or use the sample case. It is split into numbered sentences you can correct.', 'card-butter'],
            ['2', 'List the criteria', 'Type them line by line or import a real trial from ClinicalTrials.gov.', 'card-matcha'],
            ['3', 'Read the evidence', 'Each verdict shows the cited sentences, what is missing, and a downloadable report.', 'card-pink'],
          ].map(([num, title, text, tone]) => (
            <div key={num} className={`card ${tone} flex flex-col gap-2 p-5`}>
              <span className="nav-num">{num}</span>
              <h2 className="m-0 !text-lg">{title}</h2>
              <p className="m-0 text-[0.9375rem]">{text}</p>
            </div>
          ))}
        </div>
        <div className="card flex items-start gap-3 p-5">
          <TriangleAlert size={20} className="mt-0.5 shrink-0 text-[var(--rose-ink)]" aria-hidden="true" />
          <p className="m-0 text-[0.9375rem]">
            Labels follow the expert annotation scheme: <b>included / not included</b> or <b>excluded / not excluded</b>, plus <b>not enough information</b>{' '}
            and <b>not applicable</b>. A missing fact is never treated as proof either way.
          </p>
        </div>
      </div>
    );
  else if (onInspect)
    content = restoring ? null : (
      <Inspect
        key={inspected ? screening?.reportId : 'flow'}
        status={status}
        sentences={inspected ? screening!.sentences : draft.sentences}
        choices={inspected ? screening!.results : []}
        startIndex={inspected ? inspectIndex : 0}
        criterion={splitLines(draft.inclusion)[0] ?? splitLines(draft.exclusion)[0] ?? ''}
        criterionType={splitLines(draft.inclusion).length || !splitLines(draft.exclusion).length ? 'inclusion' : 'exclusion'}
        onOpenReport={openReport}
      />
    );
  else content = body;

  return (
    <div className="shell">
      <aside className="sidebar" aria-label="Main">
        <div>
          <p className="display m-0 text-[1.75rem] leading-none">CRS</p>
          <p className="hint mt-1">Clinical Records Screener</p>
        </div>
        <nav className="flex flex-col gap-4" aria-label="Sections">
          <div className="flex flex-col gap-1.5">
            <p className="eyebrow">Screening</p>
            <div className="nav-group">{steps.map(navButton)}</div>
          </div>
          <div className="flex flex-col gap-1.5">
            <p className="eyebrow">Tools</p>
            <div className="nav-group">
              {tools.map(navButton)}
              <button type="button" className="nav-item" aria-current={route === '' ? 'page' : undefined} onClick={requestHome}>
                <House size={16} aria-hidden="true" />
                Home
              </button>
            </div>
          </div>
        </nav>
        <div className="flex flex-col gap-2 md:mt-auto">
          <StatusPill status={status} />
          {status?.state === 'ready' && <p className="hint mono">running on {status.device}</p>}
        </div>
      </aside>
      <div className="flex min-w-0 flex-col">
        <p className="notice-strip m-0" role="note">
          <TriangleAlert size={16} className="shrink-0" aria-hidden="true" />
          {NOTICE}
        </p>
        <main ref={mainRef} tabIndex={-1} className="page outline-none">
          {content}
        </main>
      </div>
      {confirmHome && (
        <ConfirmDialog
          title="Leave this screening?"
          message="Going home clears the record, criteria and results on screen. Saved reports stay available in the model lab."
          confirm="Clear and go home"
          onConfirm={resetAndGoHome}
          onCancel={() => setConfirmHome(false)}
        />
      )}
    </div>
  );
}
