import { ClipboardCheck, FileText, House, ListChecks, ListOrdered, Search } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type Report, type Status } from './api';
import AppDock, { type DockEntry } from './AppDock';
import { CriteriaStep, NoteStep, Running, SentenceStep, type Draft } from './Flow';
import Inspect from './Inspect';
import LandingOrb from './LandingOrb';
import { CriterionDetail, Results, type Screening } from './Results';
import { ConfirmDialog, Notice, PageHeader } from './shared';

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
      <div className="flex max-w-md flex-col items-center gap-5">
        <Notice message={failure || 'The run failed. Try again.'} />
        <button type="button" className="btn" onClick={() => replace('criteria')}>
          Back to criteria
        </button>
      </div>
    );
  else if (route === 'results' && screening) body = <Results screening={screening} onOpen={(index) => go(`detail/${index}`)} onEdit={() => go('criteria')} />;
  else if (route.startsWith('detail/') && screening?.results[detailIndex])
    body = <CriterionDetail screening={screening} index={detailIndex} onBack={() => go('results')} onMove={(index) => replace(`detail/${index}`)} />;

  const running = route === 'running';
  const home: DockEntry = { id: 'home', label: 'Home', icon: House, onSelect: requestHome };
  const entries: DockEntry[] = inResultsArea
    ? [
        home,
        { id: 'results', label: 'Results', icon: ClipboardCheck, onSelect: () => go('results'), active: route === 'results' || route.startsWith('detail/') },
        { id: 'inspect', label: 'Inspect this result', icon: Search, onSelect: () => go(`inspect/${Math.max(detailIndex, 0)}`), active: onInspect },
      ]
    : [
        home,
        { id: 'note', label: 'Patient note', icon: FileText, onSelect: () => go('note'), active: route === 'note', disabled: running },
        { id: 'sentences', label: 'Sentences', icon: ListOrdered, onSelect: () => go('sentences'), active: route === 'sentences', disabled: running || !stepsUnlocked },
        { id: 'criteria', label: 'Criteria', icon: ListChecks, onSelect: () => go('criteria'), active: route === 'criteria' || running || route === 'failed', disabled: running || !stepsUnlocked },
        { id: 'inspect', label: 'Inspect', icon: Search, onSelect: () => go('inspect'), active: onInspect, disabled: running },
      ];

  const inspected = screening?.results[inspectIndex];

  return (
    <main ref={mainRef} tabIndex={-1} className="outline-none">
      {route === '' && (
        <div className="screen flex min-h-dvh flex-col items-center justify-center">
          <p className="display absolute top-7 left-8 m-0 text-xl font-medium tracking-wide">CRS</p>
          <LandingOrb />
          <button type="button" className="btn glass -mt-2" onClick={() => go('note')}>
            Get started
          </button>
        </div>
      )}
      {onInspect && !restoring && (
        <div className="screen min-h-dvh">
          <PageHeader status={status} />
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
        </div>
      )}
      {route !== '' && !onInspect && (
        <div className="screen flex min-h-dvh flex-col">
          <PageHeader status={status} />
          <div className={`flex flex-1 justify-center px-5 pt-8 pb-32 ${running ? 'items-center' : 'items-start md:items-center'}`}>{body}</div>
        </div>
      )}
      {route !== '' && <AppDock entries={entries} />}
      {confirmHome && (
        <ConfirmDialog
          title="Go to the home screen?"
          message="This clears the note, criteria and results on screen. Saved reports stay in Inspect."
          confirm="Go home"
          onConfirm={resetAndGoHome}
          onCancel={() => setConfirmHome(false)}
        />
      )}
    </main>
  );
}
