// Kelvin-Helmholtz hero solver: 2D incompressible Navier-Stokes on the GPU (WebGL2, no React).
//
//   du/dt + (u.grad)u = -grad p + nu lap u,   div u = 0,   dc/dt + (u.grad)c = 0 (two dyes)
//
// Domain [0, Lx] x [0, 1], periodic in x, free-slip walls in y; base flow u = tanh((y-1/2)/delta)
// seeded with its fundamental and subharmonic modes. Stable-fluids splitting per step:
// semi-Lagrangian advection of u (+ explicit viscosity), Jacobi pressure projection, MacCormack
// advection of the dyes on a 3x (2x on weak devices) finer grid. The animation recycles itself: every `cycle` seconds
// the flow is re-seeded and the old dyes cross-fade out while the new layer rolls up.

import * as S from './shaders';

export interface KHOptions {
  /** Grid height Ny of the flow (the dyes use 2 Ny). Default: 256, or 128 on small/weak devices. */
  ny?: number;
  /** Dye grid refinement over the flow grid. Default 3, or 2 on small/weak devices. */
  dyeScale?: number;
  /** Device-pixel-ratio cap for the canvas. Default: 1.5, or 1 on small/weak devices. */
  maxDpr?: number;
  /** Press and drag on the canvas to lay fluorescein (and stir gently). Default true. */
  interactive?: boolean;
  /** Seconds between re-seedings of the layer. Default 32. */
  cycle?: number;
  /** Divide the resolution when the frame rate stays below 40 fps. Default true. */
  adaptive?: boolean;
  /** Accept a software GL renderer (SwiftShader, llvmpipe...). Default false: throw instead,
   *  since a CPU rasteriser would block the main thread (the host keeps its poster). */
  allowSoftware?: boolean;
  /** Size of the canvas in CSS pixels and the device pixel ratio. Default: read from the
   *  HTMLCanvasElement; required for an OffscreenCanvas (in a worker the host sends them). */
  view?: KHView;
  /** Small or weak device (smaller grid, DPR cap). Default: viewport <= 640 px or <= 4 cores. */
  weak?: boolean;
  /** First frame drawn. */
  onReady?: () => void;
  /** The context was lost (or a later failure): the host should show its poster. */
  onFail?: (reason: string) => void;
}

export interface KHView { width(): number; height(): number; dpr(): number }
/** A pointer event relative to the canvas: x, y in CSS pixels from its top-left corner. */
export interface KHPointer { kind: 'down' | 'move' | 'up'; x: number; y: number; w: number; h: number; button: number }

export interface KHSim {
  start(): void;
  stop(): void;
  /** Advance n time steps without drawing. */
  step(n: number): void;
  render(): void;
  resize(): void;
  dispose(): void;
  /** Reader input on the canvas (the host listens to the DOM events and forwards them). */
  pointer(e: KHPointer): void;
  /** Reduced motion: advance n steps off-screen in short slices (each under ~25 ms), then draw one
   *  frame and call done. A new call or start() cancels the previous one. */
  still(n: number, done?: () => void): void;
  /** What is running, for the live readout: time since the last seeding in delta/U, grids, Re. */
  stats(): KHStats;
}

export interface KHStats { t: number; nx: number; ny: number; dyeScale: number; re: number }

type GL = WebGL2RenderingContext;

interface Target { tex: WebGLTexture; fbo: WebGLFramebuffer; w: number; h: number }
interface Pair { read: Target; write: Target; swap(): void }
interface Prog { p: WebGLProgram; u: Record<string, WebGLUniformLocation | null> }

// Physical parameters (Ly = 1, Delta U = 2).
const DELTA = 1 / 28;          // layer half-thickness: fastest-growing wavelength ~ 14 delta = 0.5
const NU = 1 / 28000;          // Re = U delta / nu = 1000
const DT = 0.002;              // CFL = U dt Ny = 0.51 at Ny = 256
const JACOBI_ITERS = 24;
const SPAN = 0.62;             // visible fraction of the domain height
const DRIFT = 0.012;           // downstream scroll of the view (uv per second of animation)
const FADE = 4;                // cross-fade seconds at each re-seeding

// Shader programs are compiled without blocking the main thread: every compile and link is issued
// first, then (KHR_parallel_shader_compile) their completion is polled once per frame, or (no
// extension) each program is checked in its own task. Checking LINK_STATUS right after linkProgram,
// as before, made the driver compile all ten programs inside one long task at start-up.
interface Pending { p: WebGLProgram; fs: WebGLShader }
function compileStart(gl: GL, vs: WebGLShader, src: string): Pending {
  const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
  gl.shaderSource(fs, src);
  gl.compileShader(fs);
  const p = gl.createProgram()!;
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  return { p, fs };
}
function compileFinish(gl: GL, { p, fs }: Pending): Prog {
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const why = gl.getShaderParameter(fs, gl.COMPILE_STATUS) ? 'link: ' + gl.getProgramInfoLog(p) : 'shader: ' + gl.getShaderInfoLog(fs);
    gl.deleteShader(fs);
    throw new Error(why);
  }
  gl.deleteShader(fs);
  const u: Prog['u'] = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS) as number;
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(p, i)!.name;
    u[name] = gl.getUniformLocation(p, name);
  }
  return { p, u };
}
// requestAnimationFrame exists in windows and in dedicated workers of current browsers; elsewhere a
// 60 Hz timer stands in.
const rAF: (cb: (t: number) => void) => number = typeof requestAnimationFrame === 'function'
  ? (cb) => requestAnimationFrame(cb) : (cb) => setTimeout(() => cb(performance.now()), 16) as unknown as number;
const cAF: (h: number) => void = typeof cancelAnimationFrame === 'function'
  ? (h) => cancelAnimationFrame(h) : (h) => clearTimeout(h);
const nextFrame = () => new Promise<void>((r) => rAF(() => r()));
const nextTask = () => new Promise<void>((r) => setTimeout(r, 0));

/** Small deterministic PRNG (mulberry32) for the seeding phases. */
function rng(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** True on phones / low-core machines: smaller grid and DPR cap. */
function weakDevice(): boolean {
  const small = typeof matchMedia === 'function' && matchMedia('(max-width: 640px)').matches;
  return small || (navigator.hardwareConcurrency || 8) <= 4;
}

/**
 * Create the solver on `canvas`, in several short tasks (context, programs, grids). Rejects if
 * WebGL2 or float render targets are unavailable, or the renderer is software (the caller then
 * keeps its static poster). `cancelled()` is polled between tasks: the creation stops (and frees
 * what it made) when the host has gone.
 */
export async function createKHSim(canvas: HTMLCanvasElement | OffscreenCanvas, opts: KHOptions = {}, cancelled: () => boolean = () => false): Promise<KHSim> {
  const gl = (canvas as HTMLCanvasElement).getContext('webgl2', {
    alpha: false, antialias: false, depth: false, stencil: false,
    premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance',
    // a software rasteriser is refused before the context exists (Chrome: SwiftShader), cheaper
    // than creating it and reading its renderer string (still checked below)
    failIfMajorPerformanceCaveat: !opts.allowSoftware,
  });
  if (!gl) throw new Error('WebGL2 unavailable');
  if (!opts.allowSoftware) {
    const info = gl.getExtension('WEBGL_debug_renderer_info');
    const name = String(gl.getParameter(info ? info.UNMASKED_RENDERER_WEBGL : gl.RENDERER));
    if (/SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(name)) throw new Error('software renderer: ' + name);
  }

  // Float formats: 32F flow if renderable and filterable, else half floats everywhere.
  const cbf = gl.getExtension('EXT_color_buffer_float');
  const lin = gl.getExtension('OES_texture_float_linear');
  const cbhf = gl.getExtension('EXT_color_buffer_half_float');
  if (!cbf && !cbhf) throw new Error('no float render targets');
  let full = !!(cbf && lin);

  const weak = opts.weak ?? weakDevice();
  const el = typeof HTMLCanvasElement !== 'undefined' && canvas instanceof HTMLCanvasElement ? canvas : null;
  const view: KHView = opts.view ?? (el ? { width: () => el.clientWidth, height: () => el.clientHeight, dpr: () => window.devicePixelRatio || 1 }
    : (() => { throw new Error('an OffscreenCanvas needs opts.view'); })());
  let ny = opts.ny ?? (weak ? 128 : 256);
  const ds = opts.dyeScale ?? (weak ? 2 : 3);
  const maxDpr = opts.maxDpr ?? (weak ? 1 : 1.5);
  const cycle = opts.cycle ?? 32;
  const interactive = opts.interactive ?? true;
  const adaptive = opts.adaptive ?? true;

  const vs = gl.createShader(gl.VERTEX_SHADER)!;
  gl.shaderSource(vs, S.VERT);
  gl.compileShader(vs);
  const SOURCES = {
    initVel: S.INIT_VEL, initDye: S.INIT_DYE, advect: S.ADVECT, correct: S.CORRECT, div: S.DIVERGENCE,
    jacobi: S.JACOBI, grad: S.GRADIENT, splat: S.SPLAT, splatDye: S.SPLAT_DYE, show: S.DISPLAY,
  };
  type Name = keyof typeof SOURCES;
  const pending = Object.fromEntries(Object.entries(SOURCES).map(([k, src]) => [k, compileStart(gl, vs, src)])) as Record<Name, Pending>;
  const abandon = () => { for (const k in pending) { gl.deleteProgram(pending[k as Name].p); gl.deleteShader(pending[k as Name].fs); } gl.deleteShader(vs); };
  const par = gl.getExtension('KHR_parallel_shader_compile') as { COMPLETION_STATUS_KHR: number } | null;
  if (par) {
    // the driver compiles on its own threads; ask once per frame whether every program is done
    for (let f = 0; f < 600; f++) {
      if (cancelled()) { abandon(); throw new Error('cancelled'); }
      if (Object.values(pending).every((q) => gl.getProgramParameter(q.p, par.COMPLETION_STATUS_KHR))) break;
      await nextFrame();
    }
  }
  const P = {} as Record<Name, Prog>;
  for (const k of Object.keys(SOURCES) as Name[]) {
    if (!par) await nextTask();               // one program per task without the extension
    if (cancelled()) { abandon(); throw new Error('cancelled'); }
    P[k] = compileFinish(gl, pending[k]);
  }
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  gl.disable(gl.BLEND);
  gl.disable(gl.DEPTH_TEST);

  const textures: Target[] = [];

  function target(w: number, h: number, fmt: 'rg' | 'r', f32: boolean, linear: boolean): Target {
    const tex = gl!.createTexture()!;
    gl!.bindTexture(gl!.TEXTURE_2D, tex);
    const ifmt = fmt === 'rg' ? (f32 ? gl!.RG32F : gl!.RG16F) : (f32 ? gl!.R32F : gl!.R16F);
    gl!.texImage2D(gl!.TEXTURE_2D, 0, ifmt, w, h, 0, fmt === 'rg' ? gl!.RG : gl!.RED,
      f32 ? gl!.FLOAT : gl!.HALF_FLOAT, null);
    const filt = linear ? gl!.LINEAR : gl!.NEAREST;
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MIN_FILTER, filt);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_MAG_FILTER, filt);
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_S, gl!.REPEAT);         // periodic x
    gl!.texParameteri(gl!.TEXTURE_2D, gl!.TEXTURE_WRAP_T, gl!.CLAMP_TO_EDGE);  // walls in y
    const fbo = gl!.createFramebuffer()!;
    gl!.bindFramebuffer(gl!.FRAMEBUFFER, fbo);
    gl!.framebufferTexture2D(gl!.FRAMEBUFFER, gl!.COLOR_ATTACHMENT0, gl!.TEXTURE_2D, tex, 0);
    if (gl!.checkFramebufferStatus(gl!.FRAMEBUFFER) !== gl!.FRAMEBUFFER_COMPLETE) {
      throw new Error('incomplete framebuffer');
    }
    const t = { tex, fbo, w, h };
    textures.push(t);
    return t;
  }
  function pair(w: number, h: number, fmt: 'rg' | 'r', f32: boolean, linear: boolean): Pair {
    const p: Pair = {
      read: target(w, h, fmt, f32, linear), write: target(w, h, fmt, f32, linear),
      swap() { const t = p.read; p.read = p.write; p.write = t; },
    };
    return p;
  }

  // ---- grid state (re-created on adaptive downgrade or a large aspect change) ----
  let lx = 2, nx = 512;
  let vel!: Pair, pres!: Pair, div!: Target;
  let dye!: Pair, ghost!: Pair, dyeHat!: Target;

  function freeGrid(): void {
    for (const t of textures) { gl!.deleteTexture(t.tex); gl!.deleteFramebuffer(t.fbo); }
    textures.length = 0;
  }
  function allocGrid(): void {
    freeGrid();
    nx = Math.round(lx * ny);
    const build = (f32: boolean) => {
      vel = pair(nx, ny, 'rg', f32, true);
      pres = pair(nx, ny, 'r', f32, false); div = target(nx, ny, 'r', f32, false);
      dye = pair(ds * nx, ds * ny, 'rg', false, true); ghost = pair(ds * nx, ds * ny, 'rg', false, true);
      dyeHat = target(ds * nx, ds * ny, 'rg', false, true);
    };
    try { build(full); } catch (e) {
      if (!full) throw e;                        // half floats failed too: give up
      freeGrid(); full = false; build(false);    // 32F not renderable here: half-float fallback
    }
  }

  /** Domain length from the canvas aspect so the view never shows a periodic repeat. */
  function wantedLx(): number {
    const a = view.width() / Math.max(1, view.height());
    return Math.min(4, Math.max(2, Math.ceil(a * SPAN * 2 + 0.25) / 2));
  }

  // ---- passes ----
  function bind(t: Target | null): void {
    gl!.bindFramebuffer(gl!.FRAMEBUFFER, t ? t.fbo : null);
    gl!.viewport(0, 0, t ? t.w : canvas.width, t ? t.h : canvas.height);
  }
  function use(pr: Prog, tex: Record<string, WebGLTexture>): void {
    gl!.useProgram(pr.p);
    let unit = 0;
    for (const k in tex) {
      gl!.activeTexture(gl!.TEXTURE0 + unit);
      gl!.bindTexture(gl!.TEXTURE_2D, tex[k]);
      gl!.uniform1i(pr.u[k], unit++);
    }
  }
  const draw = () => gl!.drawArrays(gl!.TRIANGLES, 0, 3);

  let seedCount = 7;
  /** Fresh layer: tanh profile + perturbation with new phases; smoke lines + dye filament. */
  function seed(): void {
    const r = rng(seedCount++);
    const n = Math.round(lx * 2);                           // billows: wavelength 0.5
    const k1 = 2 * Math.PI * n / lx;
    use(P.initVel, {});
    gl!.uniform2f(P.initVel.u.uRes, nx, ny);
    gl!.uniform1f(P.initVel.u.uLx, lx);
    gl!.uniform1f(P.initVel.u.uDelta, DELTA);
    gl!.uniform1f(P.initVel.u.uEps, 0.02);
    gl!.uniform4f(P.initVel.u.uMode, k1, 2 * Math.PI * r(), k1 / 2, 2 * Math.PI * r());
    gl!.uniform1f(P.initVel.u.uSub, 0.08 + 0.08 * r());
    bind(vel.read); draw();
    bind(pres.read); gl!.clearColor(0, 0, 0, 0); gl!.clear(gl!.COLOR_BUFFER_BIT);
    use(P.initDye, {});
    gl!.uniform2f(P.initDye.u.uRes, ds * nx, ds * ny);
    gl!.uniform1f(P.initDye.u.uLines, Math.round(ds * ny / 12));   // ~12 dye texels apart
    gl!.uniform1f(P.initDye.u.uDyeY, 0.5 + 1.5 * DELTA);
    bind(dye.read); draw();
  }

  /** Semi-Lagrangian step of `f` (+ viscosity when nuDt > 0), or MacCormack when `h` is given. */
  function advect(f: Pair, h: Target | null, nuDt: number): void {
    const tr = f.read;
    use(P.advect, { uVel: vel.read.tex, uSrc: tr.tex });
    gl!.uniform2f(P.advect.u.uRes, tr.w, tr.h);
    gl!.uniform2f(P.advect.u.uScale, 1 / lx, 1);
    gl!.uniform1f(P.advect.u.uDt, DT);
    gl!.uniform1f(P.advect.u.uNuDt, nuDt);
    bind(h ?? f.write); draw();
    if (!h) { f.swap(); return; }
    use(P.correct, { uVel: vel.read.tex, uSrc: tr.tex, uHat: h.tex });
    gl!.uniform2f(P.correct.u.uRes, tr.w, tr.h);
    gl!.uniform2f(P.correct.u.uScale, 1 / lx, 1);
    gl!.uniform1f(P.correct.u.uDt, DT);
    bind(f.write); draw();
    f.swap();
  }

  function project(): void {
    use(P.div, { uVel: vel.read.tex });
    gl!.uniform2i(P.div.u.uN, nx, ny);
    bind(div); draw();
    use(P.jacobi, {});
    gl!.uniform2i(P.jacobi.u.uN, nx, ny);
    for (let i = 0; i < JACOBI_ITERS; i++) {
      use(P.jacobi, { uP: pres.read.tex, uDiv: div.tex });
      bind(pres.write); draw(); pres.swap();
    }
    use(P.grad, { uP: pres.read.tex, uVel: vel.read.tex });
    gl!.uniform2i(P.grad.u.uN, nx, ny);
    bind(vel.write); draw(); vel.swap();
  }

  // Reader input, accumulated between steps: a momentum impulse (only while pressed) and the dye
  // segment travelled by the pointer since the last step.
  let pAt: [number, number] | null = null, pF: [number, number] = [0, 0];
  let dA: [number, number] | null = null, dB: [number, number] | null = null;
  function splat(): void {
    if (pAt && (pF[0] !== 0 || pF[1] !== 0)) {
      use(P.splat, { uVel: vel.read.tex });
      gl!.uniform2f(P.splat.u.uRes, nx, ny);
      gl!.uniform2f(P.splat.u.uAt, pAt[0], pAt[1]);
      gl!.uniform2f(P.splat.u.uForce, pF[0], pF[1]);
      gl!.uniform1f(P.splat.u.uLx, lx);
      gl!.uniform1f(P.splat.u.uRadius, 0.035);
      bind(vel.write); draw(); vel.swap();
      pF = [0, 0];
    }
    if (dA && dB) {
      use(P.splatDye, { uSrc: dye.read.tex });
      gl!.uniform2f(P.splatDye.u.uRes, ds * nx, ds * ny);
      gl!.uniform2f(P.splatDye.u.uA, dA[0], dA[1]);
      gl!.uniform2f(P.splatDye.u.uB, dB[0], dB[1]);
      gl!.uniform1f(P.splatDye.u.uLx, lx);
      gl!.uniform1f(P.splatDye.u.uRadius, 2.2 / (ds * ny));     // ~2 dye texels: a filament, not a blot
      bind(dye.write); draw(); dye.swap();
      dA = dB;                                                  // the next segment starts here
    }
  }

  let simT = 0;             // animation seconds of the current cycle (60 steps = 1 s)
  let steps = 0;            // time steps since the last seeding
  let fade = 0;             // seconds left in the cross-fade
  let scroll = 0;
  function stepOnce(): void {
    splat();
    advect(vel, null, NU * DT * ny * ny);    // self-advection (reads vel.read, writes vel.write)
    project();
    advect(dye, dyeHat, 0);                  // MacCormack keeps the smoke lines thin
    if (fade > 0) advect(ghost, null, 0);    // fading previous cycle: cheap semi-Lagrangian
    simT += 1 / 60; steps++; scroll += DRIFT / 60;
    if (fade > 0) fade = Math.max(0, fade - 1 / 60);
    if (simT > cycle) {                         // recycle: old dyes become the fading ghost
      const t = ghost; ghost = dye; dye = t;
      seed(); simT = 0; steps = 0; fade = FADE;
    }
  }

  let frame = 0;
  function render(): void {
    resizeCanvas();
    use(P.show, fade > 0 ? { uDye: dye.read.tex, uGhost: ghost.read.tex } : { uDye: dye.read.tex, uGhost: dye.read.tex });
    const u = P.show.u;
    gl!.uniform1f(u.uGhostMix, fade > 0 ? Math.pow(fade / FADE, 1.5) : 0);
    gl!.uniform2f(u.uView, canvas.width, canvas.height);
    gl!.uniform1f(u.uSpan, SPAN);
    gl!.uniform1f(u.uScroll, scroll);
    gl!.uniform1f(u.uLx, lx);
    gl!.uniform1f(u.uSeed, (frame++ % 97) * 7.31);
    gl!.uniform2f(u.uDyeTexel, 1 / (ds * nx), 1 / (ds * ny));
    bind(null); draw();
  }

  let dprScale = 1;          // lowered by the adaptive controller
  function resizeCanvas(): void {
    const dpr = Math.min(view.dpr(), maxDpr) * dprScale;
    const w = Math.max(1, Math.round(view.width() * dpr));
    const h = Math.max(1, Math.round(view.height() * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  }

  function rebuild(): void {
    lx = wantedLx();
    allocGrid();
    seed(); simT = 0; steps = 0; fade = 0;
  }

  // ---- animation loop + adaptive resolution ----
  let raf = 0, last = 0, acc = 0, ready = false, disposed = false;
  let meterT = 0, meterN = 0, warm = 0;
  function tick(now: number): void {
    raf = rAF(tick);
    const dt = last ? Math.min(100, now - last) : 1000 / 60;
    last = now;
    acc += dt;
    // fixed physical step per 1/60 s of wall time (at most 2 per frame, so slow GPUs slow down)
    let n = 0;
    while (acc >= 1000 / 60 - 1 && n < 2) { stepOnce(); acc -= 1000 / 60; n++; }
    if (acc > 1000 / 60) acc = 0;
    render();
    if (!ready) { ready = true; opts.onReady?.(); }
    // fps meter: 2 s windows after 1 s of warm-up; below 40 fps halve the grid (twice at most)
    if (!adaptive) return;
    if (warm < 1000) { warm += dt; return; }
    meterT += dt; meterN++;
    if (meterT >= 2000) {
      const fps = (1000 * meterN) / meterT;
      meterT = 0; meterN = 0;
      if (fps < 40 && ny > 96) {
        ny = Math.max(96, ny >> 1); dprScale *= 0.75; warm = 0;
        rebuild();
      }
    }
  }

  // Press and drag on the canvas: fluorescein along the path, plus a gentle stir. The host forwards
  // the pointer events (touch included); it sets touch-action: pan-y, so a vertical swipe still
  // scrolls the page, and it takes the pointer capture.
  let down = false, lastXY: [number, number] | null = null;
  function toUv(e: KHPointer): [number, number] | null {
    const { x, y, w, h } = e;
    if (x < 0 || y < 0 || x > w || y > h || !h) return null;
    // same mapping as the display shader (y up); x left unwrapped so a segment never jumps a period
    return [scroll + (x - 0.5 * w) / h * SPAN / lx, 0.5 + (0.5 - y / h) * SPAN];
  }
  function onDown(e: KHPointer): void {
    if (e.button !== 0) return;
    const p = toUv(e);
    if (!p) return;
    down = true; lastXY = [e.x, e.y];
    const w: [number, number] = [p[0] - Math.floor(p[0]), p[1]];
    dA = w; dB = [w[0] + 1e-4, w[1]]; pAt = w;
  }
  function onMove(e: KHPointer): void {
    if (!down) return;
    const p = toUv(e);
    if (!p) return;
    if (lastXY) {
      // velocity impulse along the pointer motion, capped (a gentle stir, not a blast)
      const k = 0.02 / e.h * 60;
      pF = [Math.max(-0.6, Math.min(0.6, pF[0] + (e.x - lastXY[0]) * k)),
            Math.max(-0.6, Math.min(0.6, pF[1] - (e.y - lastXY[1]) * k))];
    }
    lastXY = [e.x, e.y];
    const base = dA ? dA[0] : 0;
    // continue the segment from dA without crossing the period seam
    const x = p[0] - Math.floor(p[0]);
    const xu = x + Math.round(base - x);
    dB = [xu, p[1]];
    if (!dA) dA = [x, p[1]];
    pAt = [x, p[1]];
  }
  function onUp(): void {
    down = false; lastXY = null; pAt = null; dA = null; dB = null;
  }
  await nextTask();                           // grids and the first seeding in a task of their own
  if (cancelled()) {
    for (const k in P) gl.deleteProgram(P[k as Name].p);
    gl.deleteShader(vs);
    throw new Error('cancelled');
  }
  function onLost(e: Event): void {
    e.preventDefault();
    cAF(raf); raf = 0;
    opts.onFail?.('context lost');
  }
  canvas.addEventListener('webglcontextlost', onLost);

  // reduced motion: sliced stepping, the slice size adapted to its measured cost
  let stillT = 0;
  function still(n: number, done?: () => void): void {
    clearTimeout(stillT);
    let left = n, chunk = 8;
    const slice = () => {
      const t0 = performance.now();
      const k = Math.min(chunk, left);
      for (let i = 0; i < k; i++) stepOnce();
      left -= k;
      const dt = performance.now() - t0;
      chunk = Math.max(1, Math.min(120, Math.round(chunk * (dt > 0 ? 25 / dt : 2))));
      if (left > 0) stillT = setTimeout(slice, 0) as unknown as number;
      else { render(); if (!ready) { ready = true; opts.onReady?.(); } done?.(); }
    };
    stillT = setTimeout(slice, 0) as unknown as number;
  }

  resizeCanvas();
  rebuild();

  return {
    start() { if (!raf && !disposed) { clearTimeout(stillT); last = 0; acc = 0; raf = rAF(tick); } },
    stop() { cAF(raf); raf = 0; },
    pointer(e: KHPointer) { if (!interactive) return; if (e.kind === 'down') onDown(e); else if (e.kind === 'move') onMove(e); else onUp(); },
    still,
    step(n: number) { for (let i = 0; i < n; i++) stepOnce(); },
    stats() { return { t: (steps * DT) / DELTA, nx, ny, dyeScale: ds, re: Math.round(DELTA / NU) }; },
    render() { render(); if (!ready) { ready = true; opts.onReady?.(); } },
    resize() {
      resizeCanvas();
      // a much wider canvas would show the periodic repeat: re-create the domain
      if (Math.abs(wantedLx() - lx) > 0.01) rebuild();
    },
    dispose() {
      disposed = true;
      cAF(raf); raf = 0; clearTimeout(stillT);
      canvas.removeEventListener('webglcontextlost', onLost);
      freeGrid();
      for (const k in P) gl.deleteProgram(P[k as keyof typeof P].p);
      gl.deleteShader(vs);
      gl.deleteVertexArray(vao);
    },
  };
}
