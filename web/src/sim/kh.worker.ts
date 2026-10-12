// Worker side of the Kelvin-Helmholtz hero: the whole solver (WebGL2 context, shader compilation,
// every pass) runs here on an OffscreenCanvas, so the page's main thread never waits for the GPU
// driver. Protocol (messages from host.ts):
//   probe                       -> { probe: 'ok' | 'software' | 'none', renderer }
//   init { canvas, w, h, dpr, opts } -> 'created' | { error }   then 'ready' at the first frame, { fail } on context loss
//   start | stop | render | resize { w, h, dpr } | pointer { e } | still { n } (-> 'stilled') | dispose
import { createKHSim, type KHOptions, type KHPointer, type KHSim } from './kh';

type In =
  | { type: 'probe'; allowSoftware: boolean }
  | { type: 'init'; canvas: OffscreenCanvas; w: number; h: number; dpr: number; opts: Omit<KHOptions, 'view' | 'onReady' | 'onFail'> }
  | { type: 'start' | 'stop' | 'render' | 'dispose' }
  | { type: 'resize'; w: number; h: number; dpr: number }
  | { type: 'pointer'; e: KHPointer }
  | { type: 'still'; n: number };

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
let sim: KHSim | null = null, w = 1, h = 1, dpr = 1, gone = false;

/** Can this worker draw with a hardware WebGL2 context? (A throwaway 1x1 OffscreenCanvas.) */
function probe(allowSoftware: boolean): { probe: 'ok' | 'software' | 'none'; renderer: string } {
  if (typeof OffscreenCanvas === 'undefined') return { probe: 'none', renderer: '' };
  const gl = new OffscreenCanvas(1, 1).getContext('webgl2', { failIfMajorPerformanceCaveat: !allowSoftware }) as WebGL2RenderingContext | null;
  if (!gl) return { probe: allowSoftware ? 'none' : 'software', renderer: '' };
  const info = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
  gl.getExtension('WEBGL_lose_context')?.loseContext();
  const soft = /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(renderer);
  return { probe: soft && !allowSoftware ? 'software' : 'ok', renderer };
}

self.onmessage = async (ev: MessageEvent<In>) => {
  const m = ev.data;
  switch (m.type) {
    case 'probe': post(probe(m.allowSoftware)); break;
    case 'init':
      w = m.w; h = m.h; dpr = m.dpr;
      try {
        const s = await createKHSim(m.canvas, {
          ...m.opts,
          view: { width: () => w, height: () => h, dpr: () => dpr },
          onReady: () => post('ready'),
          onFail: (reason) => post({ fail: reason }),
        }, () => gone);
        if (gone) { s.dispose(); break; }
        sim = s;
        post('created');
      } catch (e) {
        post({ error: String((e as Error)?.message ?? e) });
      }
      break;
    case 'start': sim?.start(); break;
    case 'stop': sim?.stop(); break;
    case 'render': sim?.render(); break;
    case 'resize': w = m.w; h = m.h; dpr = m.dpr; sim?.resize(); break;
    case 'pointer': sim?.pointer(m.e); break;
    case 'still': sim?.still(m.n, () => post('stilled')); break;
    case 'dispose': gone = true; sim?.dispose(); sim = null; self.close(); break;
  }
};
