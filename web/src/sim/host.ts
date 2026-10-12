// Where the Kelvin-Helmholtz hero runs. Preferred: in a worker on an OffscreenCanvas
// (kh.worker.ts), so creating the WebGL context, compiling the shaders and issuing the ~30 passes
// of every frame never block the page. The canvas is handed to the worker only after the worker has
// loaded and found a hardware WebGL2 context of its own (a transferred canvas cannot come back):
//   worker probe 'ok'        -> transfer the canvas, run there
//   worker probe 'software'  -> refuse (no solver anywhere: the host shows its exact Stuart family)
//   no worker / no WebGL2 in workers / file:// / worker blocked -> the solver on the main thread
//     (kh.ts, created in short tasks)
import { createKHSim, type KHOptions, type KHPointer, type KHSim, type KHStats } from './kh';

type HostOptions = Omit<KHOptions, 'view' | 'weak'> & { weak: boolean };

const PROBE_TIMEOUT = 4000;   // ms for the worker to load and answer; then the main thread is used

export async function startKH(canvas: HTMLCanvasElement, opts: HostOptions, cancelled: () => boolean): Promise<KHSim> {
  const canWorker = typeof Worker === 'function' && typeof canvas.transferControlToOffscreen === 'function'
    && typeof location !== 'undefined' && location.protocol !== 'file:';
  if (canWorker) {
    const viaWorker = await tryWorker(canvas, opts, cancelled);
    if (viaWorker) return viaWorker;
  }
  if (cancelled()) throw new Error('cancelled');
  return createKHSim(canvas, opts, cancelled);
}

/** The solver in a worker, or null when the main thread should be used instead. Rejects when no
 *  solver should run at all (software renderer) or the worker failed after taking the canvas. */
async function tryWorker(canvas: HTMLCanvasElement, opts: HostOptions, cancelled: () => boolean): Promise<KHSim | null> {
  let worker: Worker;
  try {
    worker = new Worker(new URL('./kh.worker.ts', import.meta.url), { type: 'module' });
  } catch {
    return null;                                   // workers refused (CSP, sandbox)
  }
  const reply = <T>(timeout: number) => new Promise<T | null>((resolve) => {
    const t = setTimeout(() => done(null), timeout);
    const onMsg = (e: MessageEvent) => done(e.data as T);
    const onErr = () => done(null);
    function done(v: T | null) { clearTimeout(t); worker.removeEventListener('message', onMsg); worker.removeEventListener('error', onErr); resolve(v); }
    worker.addEventListener('message', onMsg);
    worker.addEventListener('error', onErr);
  });
  const probe = reply<{ probe: string }>(PROBE_TIMEOUT);
  worker.postMessage({ type: 'probe', allowSoftware: !!opts.allowSoftware });
  const p = await probe;
  if (!p || p.probe === 'none' || cancelled()) { worker.terminate(); if (cancelled()) throw new Error('cancelled'); return null; }
  if (p.probe === 'software') { worker.terminate(); throw new Error('software renderer'); }

  // From here the canvas belongs to the worker.
  const off = canvas.transferControlToOffscreen();
  const { onReady, onFail, ...plain } = opts;
  const created = reply<string | { error: string }>(30000);
  worker.postMessage({ type: 'init', canvas: off, w: canvas.clientWidth, h: canvas.clientHeight, dpr: window.devicePixelRatio || 1, opts: plain }, [off]);
  const c = await created;
  if (c !== 'created') { worker.terminate(); throw new Error(typeof c === 'object' && c ? c.error : 'worker did not start'); }

  let onStilled: (() => void) | undefined;
  worker.addEventListener('message', (e: MessageEvent) => {
    if (e.data === 'ready') onReady?.();
    else if (e.data === 'stilled') onStilled?.();
    else if (e.data && typeof e.data === 'object' && 'fail' in e.data) onFail?.(String(e.data.fail));
  });
  worker.addEventListener('error', () => onFail?.('worker error'));
  const send = (m: unknown) => worker.postMessage(m);
  const stats: KHStats = { t: 0, nx: 0, ny: 0, dyeScale: 0, re: 0 };   // not read by the page
  return {
    start: () => send({ type: 'start' }),
    stop: () => send({ type: 'stop' }),
    step: () => { /* the main-thread stepping API is replaced by still() */ },
    render: () => send({ type: 'render' }),
    resize: () => send({ type: 'resize', w: canvas.clientWidth, h: canvas.clientHeight, dpr: window.devicePixelRatio || 1 }),
    pointer: (e: KHPointer) => send({ type: 'pointer', e }),
    still: (n: number, done?: () => void) => { onStilled = done; send({ type: 'still', n }); },
    dispose: () => { send({ type: 'dispose' }); setTimeout(() => worker.terminate(), 1000); },
    stats: () => stats,
  };
}
