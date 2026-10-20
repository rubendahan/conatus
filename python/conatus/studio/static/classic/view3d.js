// Conatus Studio: 3D view. WebGL2, no dependency. Inlined into index.html by
// conatus.studio.page.render_page (same page for the local server and the hub).
//
// Reads the frame kinds of telemetry protocol v1 ("Trames 3D"):
//   frame3d  : 8-bit volume (strided, declared lossy) -> slices, isosurfaces, volume rendering
//   points3d : 16-bit particles / pathlines / streaklines / timelines
//   vec2d    : 8-bit 2D vector field -> LIC (line integral convolution) + streamlines
// Two views side by side: Eulerian (a frame3d field) and Lagrangian (points + dye volume).
// Colour scales of the brand charter (docs/brand/guidelines.html, section III): "smoke" (sequential),
// "signed" (rubric - black - glass, zero is black), fluorescein for the one tracked quantity.
// Everything shown here is a lossy view of the run; what was lost is printed under each view.
"use strict";
(function () {
// ------------------------------------------------------------------------------ colour scales
const hex = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
const STOPS = {
  smoke: [[0, "#0B0B0B"], [0.30, "#3A3A38"], [0.62, "#8A8A86"], [1, "#EDEBE6"]],
  signed: [[0, "#D4574A"], [0.30, "#6E2019"], [0.50, "#0B0B0B"], [0.70, "#3F5245"], [1, "#9DBBA6"]],
  dye: [[0, "#0B0B0B"], [1, "#C6F03C"]],
};
const LAB = hex("#0B0B0B"), SILVER = hex("#EDEBE6"), DYE = hex("#C6F03C"), SMOKE = hex("#8A8A86");
const RUBRIC = hex("#D4574A"), GLASS = hex("#9DBBA6");
function scaleColor(kind, p) {                  // p in [0, 1] -> [r, g, b] 0..255
  const s = STOPS[kind] || STOPS.smoke; p = Math.max(0, Math.min(1, p));
  for (let i = 1; i < s.length; i++) if (p <= s[i][0] || i === s.length - 1) {
    const a = hex(s[i - 1][1]), b = hex(s[i][1]), f = (p - s[i - 1][0]) / (s[i][0] - s[i - 1][0] || 1);
    return [0, 1, 2].map(k => a[k] + f * (b[k] - a[k]));
  }
}
// ------------------------------------------------------------------------------ decoding
function b64bytes(s) { const r = atob(s), u = new Uint8Array(r.length);
  for (let i = 0; i < r.length; i++) u[i] = r.charCodeAt(i); return u; }
const cache = new Map();                         // name -> {wall, decoded}
function decoded(fr) {
  const key = fr.run_id + "/" + fr.name, c = cache.get(key);
  if (c && c.wall === fr.wall && c.iteration === fr.iteration) return c.d;
  let d;
  if (fr.kind === "frame3d") d = { q: b64bytes(fr.data) };
  else if (fr.kind === "points3d") {
    const u8 = b64bytes(fr.data), q = new Uint16Array(u8.buffer, 0, u8.length >> 1);
    d = { q, vals: fr.values ? b64bytes(fr.values) : null };
  } else if (fr.kind === "vec2d") d = { q: b64bytes(fr.data) };
  cache.set(key, { wall: fr.wall, iteration: fr.iteration, d });
  return d;
}
// ------------------------------------------------------------------------------ small matrices
const M4 = {
  mul(a, b) { const o = new Float32Array(16);
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s; }
    return o; },
  persp(fovy, asp, n, f) { const t = 1 / Math.tan(fovy / 2), o = new Float32Array(16);
    o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f); return o; },
  look(e, c, up) {
    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const nrm = a => { const l = Math.hypot(...a) || 1; return a.map(x => x / l); };
    const cr = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const dt = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const z = nrm(sub(e, c)), x = nrm(cr(up, z)), y = cr(z, x);
    return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
      -dt(x, e), -dt(y, e), -dt(z, e), 1]); },
};
function project(mvp, p, W, H) {
  const x = mvp[0] * p[0] + mvp[4] * p[1] + mvp[8] * p[2] + mvp[12];
  const y = mvp[1] * p[0] + mvp[5] * p[1] + mvp[9] * p[2] + mvp[13];
  const w = mvp[3] * p[0] + mvp[7] * p[1] + mvp[11] * p[2] + mvp[15];
  return w > 0 ? [(x / w * 0.5 + 0.5) * W, (0.5 - y / w * 0.5) * H] : null;
}
// ------------------------------------------------------------------------------ camera (shared when synced)
function newCamera() { return { yaw: 0.75, pitch: 0.45, dist: 1.75, target: [0, 0, 0] }; }
function eye(c) { return [c.target[0] + c.dist * Math.cos(c.pitch) * Math.sin(c.yaw),
  c.target[1] + c.dist * Math.sin(c.pitch), c.target[2] + c.dist * Math.cos(c.pitch) * Math.cos(c.yaw)]; }
// ------------------------------------------------------------------------------ geometry of a frame
// box: physical extent mapped to [-s/2, s/2] with s_i = L_i / max L (true aspect ratio)
function boxOf(extent) {
  const L = extent.map(e => e[1] - e[0]), m = Math.max(...L);
  const s = L.map(l => l / m); return { ext: extent, s, min: s.map(v => -v / 2), max: s.map(v => v / 2) };
}
const toBox = (box, a, p) => (p - box.ext[a][0]) / (box.ext[a][1] - box.ext[a][0]) * box.s[a] - box.s[a] / 2;
// fractional sample index -> physical coordinate along an axis (coords are always sent)
function idxToPhys(coords, f) { const n = coords.length; if (n === 1) return coords[0];
  const i = Math.max(0, Math.min(n - 2, Math.floor(f))), t = f - i; return coords[i] + t * (coords[i + 1] - coords[i]); }
// physical coordinate -> texture coordinate (i + 0.5) / n, piecewise linear in the sample positions
function physToTex(coords, p) { const n = coords.length;
  if (n === 1 || p <= coords[0]) return 0.5 / n;
  if (p >= coords[n - 1]) return (n - 0.5) / n;
  let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (coords[m] <= p) lo = m; else hi = m; }
  return (lo + (p - coords[lo]) / (coords[hi] - coords[lo]) + 0.5) / n; }
// ------------------------------------------------------------------------------ display normalization
// Declared client-side clipping for intermittent fields (Q, lambda2: most voxels near 0): the scale
// is a percentile of the received 8-bit volume, computed from its 256-bin histogram. signed: A =
// pct-th percentile of |v| over the voxels of the selected sign(s); smoke: [100-pct, pct] percentiles.
function clipScale(fr, q, pct, sign) {
  const h = new Float64Array(256); for (let i = 0; i < q.length; i++) h[q[i]]++;
  const val = i => fr.min + (fr.max - fr.min) * i / 255;
  if (fr.scale !== "signed") {
    let c = 0, lo = fr.min, hi = fr.max, got = false; const N = q.length;
    for (let i = 0; i < 256; i++) { c += h[i]; if (!got && c >= N * (1 - pct / 100)) { lo = val(i); got = true; } if (c >= N * pct / 100) { hi = val(i); break; } }
    return { lo, hi: hi > lo ? hi : fr.max };
  }
  const pairs = []; let N = 0;
  for (let i = 0; i < 256; i++) { const v = val(i); if (!h[i] || (sign === "+" && v <= 0) || (sign === "-" && v >= 0)) continue;
    pairs.push([Math.abs(v), h[i]]); N += h[i]; }
  pairs.sort((a, b) => a[0] - b[0]);
  let c = 0, A = 0; for (const [a, n] of pairs) { c += n; A = a; if (c >= N * pct / 100) break; }
  return { A: A || Math.max(Math.abs(fr.min), Math.abs(fr.max)) || 1 };
}
// ------------------------------------------------------------------------------ isosurface
// Marching tetrahedra (each cell cut into 6 tetrahedra around its main diagonal: no ambiguous
// case, no 256-entry table) on the 8-bit volume as received: a lossy isosurface, declared.
const CORNER = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
const TETS = [[0,5,1,6],[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6]];
function isosurface(q, sh, level, coords, box) {
  const [n0, n1, n2] = sh, at = (i, j, k) => q[(i * n1 + j) * n2 + k];
  const pos = [], nor = [];
  const grad = (i, j, k) => {
    const g = (a, b, fa, fb) => (fb - fa) / Math.max(1, b - a);
    const i0 = Math.max(0, i - 1), i1 = Math.min(n0 - 1, i + 1), j0 = Math.max(0, j - 1), j1 = Math.min(n1 - 1, j + 1);
    const k0 = Math.max(0, k - 1), k1 = Math.min(n2 - 1, k + 1);
    return [g(i0, i1, at(i0, j, k), at(i1, j, k)), g(j0, j1, at(i, j0, k), at(i, j1, k)), g(k0, k1, at(i, j, k0), at(i, j, k1))];
  };
  const cv = new Float32Array(8), cp = [], cg = [];
  const emit = (a, b) => {                       // vertex on edge a-b of the current cell
    const t = (level - cv[a]) / (cv[b] - cv[a]);
    const p = [0, 1, 2].map(d => cp[a][d] + t * (cp[b][d] - cp[a][d]));
    const g = [0, 1, 2].map(d => cg[a][d] + t * (cg[b][d] - cg[a][d]));
    for (let d = 0; d < 3; d++) pos.push(toBox(box, d, idxToPhys(coords[d], p[d])));
    const l = Math.hypot(...g) || 1; nor.push(-g[0] / l, -g[1] / l, -g[2] / l);
  };
  for (let i = 0; i < n0 - 1; i++) for (let j = 0; j < n1 - 1; j++) for (let k = 0; k < n2 - 1; k++) {
    let mn = 256, mx = -1;
    for (let c = 0; c < 8; c++) { const v = at(i + CORNER[c][0], j + CORNER[c][1], k + CORNER[c][2]);
      cv[c] = v; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mn >= level || mx < level) continue;
    for (let c = 0; c < 8; c++) { cp[c] = [i + CORNER[c][0], j + CORNER[c][1], k + CORNER[c][2]];
      cg[c] = grad(cp[c][0], cp[c][1], cp[c][2]); }
    for (const T of TETS) {
      const ins = T.filter(c => cv[c] >= level), out = T.filter(c => cv[c] < level);
      if (ins.length === 1 || ins.length === 3) {
        const [a, bs] = ins.length === 1 ? [ins[0], out] : [out[0], ins];
        emit(a, bs[0]); emit(a, bs[1]); emit(a, bs[2]);
      } else if (ins.length === 2) {
        const [a, b] = ins, [c, d] = out;
        emit(a, c); emit(a, d); emit(b, d); emit(a, c); emit(b, d); emit(b, c);
      }
    }
  }
  return { pos: new Float32Array(pos), nor: new Float32Array(nor), n: pos.length / 3 };
}
// ------------------------------------------------------------------------------ shaders
const VS_BOX = `#version 300 es
in vec3 aPos; uniform mat4 uMVP; out vec3 vPos;
void main() { vPos = aPos; gl_Position = uMVP * vec4(aPos, 1.0); }`;
const FS_VOL = `#version 300 es
precision highp float; precision highp sampler3D;
in vec3 vPos; out vec4 o;
uniform sampler3D uVol; uniform sampler2D uTF; uniform sampler2D uMap;
uniform vec3 uCam, uMin, uMax; uniform int uSteps; uniform float uOpacity;
vec3 tex(vec3 p) { vec3 u = clamp((p - uMin) / (uMax - uMin), 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0;
  return vec3(texture(uMap, vec2(u.z, 2.5 / 3.0)).r, texture(uMap, vec2(u.y, 1.5 / 3.0)).r,
              texture(uMap, vec2(u.x, 0.5 / 3.0)).r); }
void main() {
  vec3 d = normalize(vPos - uCam);
  vec3 t0 = (uMin - uCam) / d, t1 = (uMax - uCam) / d;
  vec3 tn = min(t0, t1), tf = max(t0, t1);
  float a = max(max(max(tn.x, tn.y), tn.z), 0.0), b = min(min(tf.x, tf.y), tf.z);
  if (b <= a) discard;
  float dt = (b - a) / float(uSteps), ref = length(uMax - uMin) / 200.0;
  vec4 acc = vec4(0.0);
  for (int i = 0; i < 1024; i++) {
    if (i >= uSteps || acc.a > 0.97) break;
    vec3 p = uCam + d * (a + (float(i) + 0.5) * dt);
    vec4 c = texture(uTF, vec2(texture(uVol, tex(p)).r, 0.5));
    float al = 1.0 - pow(1.0 - clamp(c.a * uOpacity, 0.0, 0.999), dt / ref);   // opacity per step length
    acc.rgb += (1.0 - acc.a) * al * c.rgb; acc.a += (1.0 - acc.a) * al;
  }
  o = acc;
}`;
const FS_SLICE = `#version 300 es
precision highp float; precision highp sampler3D;
in vec3 vPos; out vec4 o;
uniform sampler3D uVol; uniform sampler2D uTF; uniform sampler2D uMap; uniform vec3 uMin, uMax;
void main() { vec3 u = clamp((vPos - uMin) / (uMax - uMin), 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0;
  vec3 t = vec3(texture(uMap, vec2(u.z, 2.5 / 3.0)).r, texture(uMap, vec2(u.y, 1.5 / 3.0)).r,
                texture(uMap, vec2(u.x, 0.5 / 3.0)).r);
  o = vec4(texture(uTF, vec2(texture(uVol, t).r, 0.5)).rgb, 1.0); }`;
const VS_MESH = `#version 300 es
in vec3 aPos; in vec3 aNor; uniform mat4 uMVP; out vec3 vPos; out vec3 vNor;
void main() { vPos = aPos; vNor = aNor; gl_Position = uMVP * vec4(aPos, 1.0); }`;
const FS_MESH = `#version 300 es
precision highp float; in vec3 vPos; in vec3 vNor; out vec4 o; uniform vec3 uCam; uniform vec3 uColor;
void main() { vec3 n = normalize(vNor), v = normalize(uCam - vPos);
  float l = 0.22 + 0.78 * abs(dot(n, v)); o = vec4(uColor * l, 1.0); }`;
const VS_PT = `#version 300 es
in vec3 aPos; uniform mat4 uMVP; uniform float uSize;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); gl_PointSize = uSize; }`;
const FS_PT = `#version 300 es
precision mediump float; out vec4 o; uniform vec4 uColor; uniform int uRound;
void main() { if (uRound == 1 && length(gl_PointCoord - 0.5) > 0.5) discard; o = uColor; }`;
// ------------------------------------------------------------------------------ one WebGL2 view
class GLView {
  constructor(canvas, overlay, ctrl) {
    this.cv = canvas; this.ov = overlay; this.ctrl = ctrl;
    const gl = this.gl = canvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false });
    if (!gl) return;
    this.prog = {
      vol: this.program(VS_BOX, FS_VOL), slice: this.program(VS_BOX, FS_SLICE),
      mesh: this.program(VS_MESH, FS_MESH), pt: this.program(VS_PT, FS_PT) };
    this.buf = {}; this.vol = null; this.box = boxOf([[0, 1], [0, 1], [0, 1]]);
    this.layers = [];                           // [{kind: "lines"|"points"|"mesh", ...}]
    this.cube = gl.createBuffer();
  }
  program(vs, fs) {
    const gl = this.gl, p = gl.createProgram();
    for (const [t, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s); }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    p.u = n => gl.getUniformLocation(p, n); return p;
  }
  // volume: 8-bit 3D texture + transfer function + physical -> texture maps (one row per axis)
  setVolume(fr, q, scale) {
    const gl = this.gl, [n0, n1, n2] = fr.shape;
    if (!this.vol) this.vol = { tex: gl.createTexture(), tf: gl.createTexture(), map: gl.createTexture() };
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindTexture(gl.TEXTURE_3D, this.vol.tex);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, n2, n1, n0, 0, gl.RED, gl.UNSIGNED_BYTE, q);
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR],
      [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE]])
      gl.texParameteri(gl.TEXTURE_3D, k, v);
    this.setTF(fr, scale, this.clip);
    const map = new Float32Array(256 * 3);
    for (let a = 0; a < 3; a++) for (let i = 0; i < 256; i++) {
      const p = fr.extent[a][0] + (fr.extent[a][1] - fr.extent[a][0]) * i / 255;
      map[a * 256 + i] = physToTex(fr.coords[a], p); }
    gl.bindTexture(gl.TEXTURE_2D, this.vol.map);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R16F, 256, 3, 0, gl.RED, gl.FLOAT, map);
    this.tex2dParams(gl.LINEAR);
    this.box = boxOf(fr.extent); this.volFrame = fr;
  }
  // 256 x 1 RGBA: colour and opacity of each 8-bit level; clip = {A, sign} (signed) or {lo, hi}
  // (smoke), the declared client-side display scale (clipScale); none: the frame's own min/max
  setTF(fr, scale, clip) {
    const gl = this.gl, tf = new Uint8Array(256 * 4);
    const A = (clip && clip.A) || Math.max(Math.abs(fr.min), Math.abs(fr.max)) || 1, sign = (clip && clip.sign) || "±";
    const lo = clip && clip.lo !== undefined ? clip.lo : fr.min, hi = clip && clip.hi !== undefined ? clip.hi : fr.max;
    for (let i = 0; i < 256; i++) {
      const v = fr.min + (fr.max - fr.min) * i / 255;
      let p, a;
      if (scale === "signed") { const r = Math.max(-1, Math.min(1, v / A)); p = 0.5 + 0.5 * r; a = r * r;
        if ((sign === "+" && v < 0) || (sign === "-" && v > 0)) a = 0; }
      else if (scale === "dye") { p = i / 255; a = p; }
      else { p = Math.max(0, Math.min(1, (v - lo) / ((hi - lo) || 1))); a = p * p; }
      const c = scale === "dye" ? DYE.map(x => x * Math.min(1, 0.25 + p)) : scaleColor(scale, p);
      tf.set([c[0], c[1], c[2], 255 * a], 4 * i); }
    const gl2 = gl; gl2.bindTexture(gl.TEXTURE_2D, this.vol.tf);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, tf);
    this.tex2dParams(gl.LINEAR);
  }
  tex2dParams(f) { const gl = this.gl;
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); }
  clearVolume() { this.volFrame = null; }
  array(key, data) { const gl = this.gl; let b = this.buf[key];
    if (!b) b = this.buf[key] = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW); return b; }
  attrib(prog, name, buf, n) { const gl = this.gl, l = gl.getAttribLocation(prog, name);
    if (l < 0) return; gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, n, gl.FLOAT, false, 0, 0); }
  bindVol(p) { const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_3D, this.vol.tex); gl.uniform1i(p.u("uVol"), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.vol.tf); gl.uniform1i(p.u("uTF"), 1);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.vol.map); gl.uniform1i(p.u("uMap"), 2);
    gl.uniform3fv(p.u("uMin"), this.box.min); gl.uniform3fv(p.u("uMax"), this.box.max); }
  draw(cam) {
    const gl = this.gl; if (!gl) return;
    const t0 = performance.now();
    const dpr = Math.min(window.devicePixelRatio || 1, this.ctrl.mobile ? 1.5 : 2);
    const W = this.cv.clientWidth, H = this.cv.clientHeight;
    if (this.cv.width !== Math.round(W * dpr) || this.cv.height !== Math.round(H * dpr)) {
      this.cv.width = Math.round(W * dpr); this.cv.height = Math.round(H * dpr);
      this.ov.width = this.cv.width; this.ov.height = this.cv.height; }
    gl.viewport(0, 0, this.cv.width, this.cv.height);
    gl.clearColor(LAB[0] / 255, LAB[1] / 255, LAB[2] / 255, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const E = eye(cam), mvp = M4.mul(M4.persp(0.62, W / Math.max(1, H), 0.02, 50), M4.look(E, cam.target, [0, 1, 0]));
    this.mvp = mvp;
    gl.enable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    const box = this.box, c = this.ctrl;
    // slices (opaque)
    if (this.volFrame && c.slices) {
      const p = this.prog.slice; gl.useProgram(p); gl.uniformMatrix4fv(p.u("uMVP"), false, mvp); this.bindVol(p);
      const quads = [];
      for (let a = 0; a < 3; a++) if (c.slices[a].on) {
        const e = box.ext[a], v = toBox(box, a, e[0] + c.slices[a].f * (e[1] - e[0]));
        const b = (a + 1) % 3, d = (a + 2) % 3, P = (s, t) => { const r = [0, 0, 0]; r[a] = v; r[b] = s; r[d] = t; return r; };
        const [b0, b1, d0, d1] = [box.min[b], box.max[b], box.min[d], box.max[d]];
        quads.push(...P(b0, d0), ...P(b1, d0), ...P(b1, d1), ...P(b0, d0), ...P(b1, d1), ...P(b0, d1)); }
      if (quads.length) { this.attrib(p, "aPos", this.array("slice", new Float32Array(quads)), 3);
        gl.drawArrays(gl.TRIANGLES, 0, quads.length / 3); }
    }
    // meshes (isosurfaces)
    for (const L of this.layers.filter(l => l.kind === "mesh" && l.n)) {
      const p = this.prog.mesh; gl.useProgram(p); gl.uniformMatrix4fv(p.u("uMVP"), false, mvp);
      gl.uniform3fv(p.u("uCam"), E); gl.uniform3fv(p.u("uColor"), L.color.map(x => x / 255));
      this.attrib(p, "aPos", this.array(L.key + "p", L.pos), 3); this.attrib(p, "aNor", this.array(L.key + "n", L.nor), 3);
      gl.drawArrays(gl.TRIANGLES, 0, L.n);
      gl.disableVertexAttribArray(gl.getAttribLocation(p, "aNor"));
    }
    // box edges, lines and points
    const p = this.prog.pt; gl.useProgram(p); gl.uniformMatrix4fv(p.u("uMVP"), false, mvp);
    const [x0, y0, z0] = box.min, [x1, y1, z1] = box.max, ed = [];
    for (const [a, b] of [[[x0,y0,z0],[x1,y0,z0]],[[x0,y1,z0],[x1,y1,z0]],[[x0,y0,z1],[x1,y0,z1]],[[x0,y1,z1],[x1,y1,z1]],
      [[x0,y0,z0],[x0,y1,z0]],[[x1,y0,z0],[x1,y1,z0]],[[x0,y0,z1],[x0,y1,z1]],[[x1,y0,z1],[x1,y1,z1]],
      [[x0,y0,z0],[x0,y0,z1]],[[x1,y0,z0],[x1,y0,z1]],[[x0,y1,z0],[x0,y1,z1]],[[x1,y1,z0],[x1,y1,z1]]]) ed.push(...a, ...b);
    this.attrib(p, "aPos", this.array("box", new Float32Array(ed)), 3);
    gl.uniform4f(p.u("uColor"), SMOKE[0] / 255, SMOKE[1] / 255, SMOKE[2] / 255, 1); gl.uniform1i(p.u("uRound"), 0);
    gl.drawArrays(gl.LINES, 0, ed.length / 3);
    for (const L of this.layers.filter(l => l.kind === "lines" || l.kind === "points")) {
      this.attrib(p, "aPos", this.array(L.key, L.pos), 3);
      gl.uniform4f(p.u("uColor"), L.color[0] / 255, L.color[1] / 255, L.color[2] / 255, 1);
      gl.uniform1f(p.u("uSize"), L.size * dpr); gl.uniform1i(p.u("uRound"), 1);
      gl.drawArrays(L.kind === "lines" ? gl.LINES : gl.POINTS, 0, L.pos.length / 3);
    }
    // volume (ray marching, composited over what is drawn; no depth test against geometry)
    if (this.volFrame && c.volume) {
      const q = this.prog.vol; gl.useProgram(q); gl.uniformMatrix4fv(q.u("uMVP"), false, mvp); this.bindVol(q);
      gl.uniform3fv(q.u("uCam"), E); gl.uniform1i(q.u("uSteps"), c.mobile ? 64 : 192);
      gl.uniform1f(q.u("uOpacity"), c.opacity);
      const cube = [], F = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[3,2,6,7],[0,3,7,4],[1,2,6,5]];
      const V = i => [i & 1 ? x1 : x0, i & 2 ? y1 : y0, i & 4 ? z1 : z0], ord = [0, 1, 3, 2, 4, 5, 7, 6];
      const ctr = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2];
      for (const f of F) { let v = f.map(i => V(ord[i]));
        // outward counter-clockwise winding on every face (front-face culling needs one convention)
        const e1 = [0, 1, 2].map(k => v[1][k] - v[0][k]), e2 = [0, 1, 2].map(k => v[2][k] - v[0][k]);
        const n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
        if (n[0] * (v[0][0] - ctr[0]) + n[1] * (v[0][1] - ctr[1]) + n[2] * (v[0][2] - ctr[2]) < 0) v = [v[0], v[3], v[2], v[1]];
        cube.push(...v[0], ...v[1], ...v[2], ...v[0], ...v[2], ...v[3]); }
      this.attrib(q, "aPos", this.array("cube", new Float32Array(cube)), 3);
      gl.disable(gl.DEPTH_TEST); gl.enable(gl.CULL_FACE); gl.cullFace(gl.FRONT);    // back faces: works inside the box too
      gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.TRIANGLES, 0, cube.length / 3);
      gl.disable(gl.CULL_FACE); gl.disable(gl.BLEND);
    }
    this.overlay(W, H, dpr);
    this.ms = performance.now() - t0;
  }
  overlay(W, H, dpr) {                          // axis names at the box edges, drawn in 2D
    const g = this.ov.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    g.font = "12px ui-monospace, Menlo, monospace"; g.fillStyle = "rgba(237,235,230,.85)";
    const b = this.box, names = (this.axes || ["x", "y", "z"]);
    for (let a = 0; a < 3; a++) { const p = [...b.min]; p[a] = b.max[a] * 1.12;
      const s = project(this.mvp, p, W, H); if (s) g.fillText(names[a], s[0], s[1]); }
  }
}
// ------------------------------------------------------------------------------ LIC (2D, CPU)
// Cabral & Leedom 1993: grey level = average of a white noise along the streamline through each
// pixel (RK2, 0.5 px steps, box kernel of +-L px), modulated by |u| on the smoke scale.
function lic(fr, q, W, H, Lpx) {
  const [nx, ny] = fr.shape, amp = fr.amp || 1, N = nx * ny, E = fr.extent;
  const sx = W / (E[0][1] - E[0][0]), sy = H / (E[1][1] - E[1][0]);   // pixels per unit length
  const U = new Float32Array(N), V = new Float32Array(N);
  for (let i = 0; i < N; i++) { U[i] = (q[i] - 127.5) / 127.5; V[i] = (q[N + i] - 127.5) / 127.5; }
  const vel = (x, y) => {                       // bilinear, pixel coords -> grid (x: axis 0, y: axis 1 up)
    const gx = Math.max(0, Math.min(nx - 1.001, x / W * (nx - 1))), gy = Math.max(0, Math.min(ny - 1.001, (H - 1 - y) / H * (ny - 1)));
    const i = gx | 0, j = gy | 0, fx = gx - i, fy = gy - j, k = i * ny + j;
    const u = (1 - fx) * ((1 - fy) * U[k] + fy * U[k + 1]) + fx * ((1 - fy) * U[k + ny] + fy * U[k + ny + 1]);
    const v = (1 - fx) * ((1 - fy) * V[k] + fy * V[k + 1]) + fx * ((1 - fy) * V[k + ny] + fy * V[k + ny + 1]);
    return [u * sx, -v * sy]; };             // direction in pixels (screen y points down)
  let seed = 34; const noise = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) { seed = (seed * 1664525 + 1013904223) >>> 0; noise[i] = seed / 4294967296; }
  const img = new ImageData(W, H), steps = Math.round(2 * Lpx);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = noise[y * W + x], n = 1;
    for (const dir of [1, -1]) { let px = x + 0.5, py = y + 0.5;
      for (let k = 0; k < steps; k++) {
        let [u, v] = vel(px, py); let m = Math.hypot(u, v); if (m < 1e-9) break;
        const hx = px + dir * 0.25 * u / m, hy = py + dir * 0.25 * v / m;
        [u, v] = vel(hx, hy); m = Math.hypot(u, v); if (m < 1e-9) break;
        px += dir * 0.5 * u / m; py += dir * 0.5 * v / m;
        if (px < 0 || py < 0 || px >= W || py >= H) break;
        s += noise[(py | 0) * W + (px | 0)]; n++; } }
    const [u, v] = vel(x + 0.5, y + 0.5), mag = Math.min(1, Math.hypot(u / sx, v / sy));
    const c = scaleColor("smoke", 0.25 + 0.75 * mag), l = 0.35 + 1.3 * (s / n - 0.5) * 1.6, p = 4 * (y * W + x);
    img.data[p] = c[0] * l; img.data[p + 1] = c[1] * l; img.data[p + 2] = c[2] * l; img.data[p + 3] = 255; }
  return { img, vel, amp };
}
// ------------------------------------------------------------------------------ the panel
const V3 = window.View3D = { stats: {} };
const el = (h) => { const d = document.createElement("div"); d.innerHTML = h.trim(); return d.firstChild; };
const esc3 = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const ctrl = { slices: [{ on: true, f: 0.5 }, { on: false, f: 0.5 }, { on: false, f: 0.5 }], iso: false, isoF: 0.5, clipPct: 99, sign: "±", signUser: false,
  volume: false, opacity: 0.6, sync: true, mobile: matchMedia("(max-width: 600px), (pointer: coarse)").matches,
  field: null, dye: "", lines: true, stream: true };
let built = false, views = [], cams = [newCamera(), newCamera()], runKey = null, last = {}, raf = 0;
function build(panel) {
  const box = panel.querySelector("#view3d");
  box.innerHTML = `<div id="v3three">
  <div class="row v3ctl">
    <label class="muted">field <select class="in" id="v3field"></select></label>
    <label class="muted"><input type="checkbox" id="v3sx" checked> x</label><input type="range" id="v3fx" min="0" max="1" step="0.01" value="0.5" aria-label="x slice">
    <label class="muted"><input type="checkbox" id="v3sy"> y</label><input type="range" id="v3fy" min="0" max="1" step="0.01" value="0.5" aria-label="y slice">
    <label class="muted"><input type="checkbox" id="v3sz"> z</label><input type="range" id="v3fz" min="0" max="1" step="0.01" value="0.5" aria-label="z slice">
  </div>
  <div class="row v3ctl">
    <label class="muted"><input type="checkbox" id="v3iso"> isosurface</label><input type="range" id="v3isof" min="0.02" max="0.98" step="0.01" value="0.5" aria-label="isosurface level (fraction of the clipped scale)">
    <span class="mono muted" id="v3isov"></span>
    <label class="muted">sign <select class="in" id="v3sign"><option value="±">±</option><option value="+">+</option><option value="-">−</option></select></label>
    <label class="muted">clip <input type="range" id="v3clip" min="90" max="99.9" step="0.1" value="99" aria-label="clip percentile"></label><span class="mono muted" id="v3clipv"></span>
    <label class="muted"><input type="checkbox" id="v3vol"> volume</label><input type="range" id="v3op" min="0.05" max="1" step="0.05" value="0.6" aria-label="volume opacity">
    <label class="muted">dye <select class="in" id="v3dye"></select></label>
    <label class="muted"><input type="checkbox" id="v3sync" checked> sync cameras</label>
    <button id="v3reset" title="reset both cameras (double tap on a view)">Reset view</button>
  </div>
  <div class="v3grid">
    <figure class="v3fig"><div class="v3wrap"><canvas class="v3c" id="v3e"></canvas><canvas class="v3o"></canvas></div>
      <figcaption class="mono muted" id="v3ecap">Eulerian</figcaption></figure>
    <figure class="v3fig"><div class="v3wrap"><canvas class="v3c" id="v3l"></canvas><canvas class="v3o"></canvas></div>
      <figcaption class="mono muted" id="v3lcap">Lagrangian</figcaption></figure>
  </div></div>
  <div id="v3two" hidden><h2 style="margin-top:12px">2D: Eulerian (LIC) and Lagrangian</h2>
    <div class="row v3ctl"><label class="muted">vector field <select class="in" id="v3vec"></select></label>
      <label class="muted"><input type="checkbox" id="v3st" checked> streamlines</label>
      <label class="muted">under the particles <select class="in" id="v3bg"></select></label></div>
    <div class="v3grid">
      <figure class="v3fig"><canvas id="v3lic" style="width:100%;display:block"></canvas><figcaption class="mono muted" id="v3liccap"></figcaption></figure>
      <figure class="v3fig"><canvas id="v3lag2" style="width:100%;display:block;background:#0B0B0B"></canvas><figcaption class="mono muted" id="v3lag2cap"></figcaption></figure>
    </div></div>
  <div class="note" id="v3note"></div>`;
  const st = document.createElement("style");
  st.textContent = `.v3grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(340px,100%),1fr));gap:12px;margin-top:8px}
  .v3fig{margin:0;min-width:0}.v3wrap{position:relative;width:100%;height:min(62vh,420px);background:#0B0B0B;border-radius:4px;overflow:hidden}
  .v3c,.v3o{position:absolute;inset:0;width:100%;height:100%;display:block}.v3c{touch-action:none;cursor:grab}.v3o{pointer-events:none}
  .v3ctl input[type=range]{width:min(120px,28vw)}.v3fig figcaption{margin-top:4px;overflow-wrap:anywhere}
  @media (max-width:600px){.v3wrap{height:52vh}}`;
  document.head.appendChild(st);
  const on = (id, ev, f) => box.querySelector("#" + id).addEventListener(ev, f);
  "xyz".split("").forEach((a, i) => { on("v3s" + a, "change", e => { ctrl.slices[i].on = e.target.checked; redraw(); });
    on("v3f" + a, "input", e => { ctrl.slices[i].f = +e.target.value; redraw(); }); });
  on("v3field", "change", e => { ctrl.field = e.target.value; ctrl.signUser = false; last = {}; V3.update(V3.R, true); });
  on("v3dye", "change", e => { ctrl.dye = e.target.value; last = {}; V3.update(V3.R, true); });
  on("v3vec", "change", () => { last.lic = null; V3.update(V3.R, true); });
  on("v3st", "change", () => { last.lic = null; V3.update(V3.R, true); });
  on("v3bg", "change", () => { last.lag2 = null; V3.update(V3.R, true); });
  on("v3iso", "change", e => { ctrl.iso = e.target.checked; last.iso = null; V3.update(V3.R, true); });
  on("v3isof", "change", e => { ctrl.isoF = +e.target.value; last.iso = null; V3.update(V3.R, true); });
  on("v3sign", "change", e => { ctrl.sign = e.target.value; ctrl.signUser = true; last.F = null; last.iso = null; V3.update(V3.R, true); });
  on("v3clip", "change", e => { ctrl.clipPct = +e.target.value; last.F = null; last.iso = null; V3.update(V3.R, true); });
  on("v3vol", "change", e => { ctrl.volume = e.target.checked; redraw(); });
  on("v3op", "input", e => { ctrl.opacity = +e.target.value; redraw(); });
  on("v3sync", "change", e => { ctrl.sync = e.target.checked; if (ctrl.sync) cams[1] = cams[0]; else cams[1] = { ...cams[0], target: [...cams[0].target] }; redraw(); });
  on("v3reset", "click", () => { cams[0] = newCamera(); cams[1] = ctrl.sync ? cams[0] : newCamera(); redraw(); });
  cams[1] = cams[0];
  const cvs = [box.querySelector("#v3e"), box.querySelector("#v3l")];
  views = cvs.map((c, i) => { let v; try { v = new GLView(c, c.nextElementSibling, ctrl); } catch (err) { v = { gl: null, err }; }
    if (v.gl) gestures(c, i); return v; });
  if (!views[0].gl) box.querySelector("#v3note").innerHTML = `<span class="warnbox">WebGL2 is not available in this browser
    (${esc3(views[0].err || "no context")}): 3D rendering is off; mid-plane slices are drawn in 2D instead.</span>`;
  window.addEventListener("resize", redraw);
  built = true;
}
// mouse and touch: 1 pointer rotates (shift or right button pans), 2 pointers pinch-zoom and pan,
// wheel zooms, double tap / double click resets
function gestures(c, i) {
  const P = new Map(); let lastTap = 0, pinch = null;
  c.addEventListener("contextmenu", e => e.preventDefault());
  c.addEventListener("pointerdown", e => { try { c.setPointerCapture(e.pointerId); } catch (err) { /* synthetic or ended pointer */ } P.set(e.pointerId, [e.clientX, e.clientY, e.button, e.shiftKey]);
    if (P.size === 1) { const now = Date.now(); if (now - lastTap < 300) { cams[i] = newCamera(); if (ctrl.sync) cams[1 - i] = cams[i]; redraw(); } lastTap = now; }
    pinch = null; });
  const end = e => { P.delete(e.pointerId); pinch = null; };
  c.addEventListener("pointerup", end); c.addEventListener("pointercancel", end);
  c.addEventListener("pointermove", e => {
    if (!P.has(e.pointerId)) return;
    const cam = cams[i], [px, py, btn, sh] = P.get(e.pointerId), dx = e.clientX - px, dy = e.clientY - py;
    P.set(e.pointerId, [e.clientX, e.clientY, btn, sh]);
    if (P.size === 1) {
      if (btn === 2 || sh) pan(cam, dx, dy, c);
      else { cam.yaw -= dx * 0.008; cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + dy * 0.008)); }
    } else if (P.size === 2) {
      const [a, b] = [...P.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]), m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      if (pinch) { cam.dist = Math.max(0.3, Math.min(12, cam.dist * pinch.d / Math.max(1, d))); pan(cam, m[0] - pinch.m[0], m[1] - pinch.m[1], c); }
      pinch = { d, m };
    }
    redraw(); });
  c.addEventListener("wheel", e => { e.preventDefault(); cams[i].dist = Math.max(0.3, Math.min(12, cams[i].dist * Math.exp(e.deltaY * 0.001))); redraw(); }, { passive: false });
}
function pan(cam, dx, dy, c) {
  const s = cam.dist / c.clientHeight * 0.75, cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), sp = Math.sin(cam.pitch);
  const right = [cy, 0, -sy], up = [-sp * sy, Math.cos(cam.pitch), -sp * cy];
  for (let k = 0; k < 3; k++) cam.target[k] += (-dx * right[k] + dy * up[k]) * s;
}
function redraw() { if (raf) return; raf = requestAnimationFrame(() => { raf = 0;
  const t0 = performance.now();
  views.forEach((v, i) => { if (v.gl) v.draw(cams[i]); });
  V3.stats.frame_ms = +(performance.now() - t0).toFixed(2); }); }
function options(sel, names, keep) {
  const cur = sel.value, key = names.join("|");
  if (sel.dataset.k !== key) { sel.innerHTML = names.map(n => `<option value="${esc3(n)}">${esc3(n || "none")}</option>`).join(""); sel.dataset.k = key; }
  if (names.includes(keep ?? cur)) sel.value = keep ?? cur;
  return sel.value;
}
function fmtv(v) { return typeof v !== "number" ? String(v) : (v === 0 ? "0" : (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3) ? v.toExponential(2) : v.toPrecision(4)); }
function lossy(fr) { return `${fr.shape.join("×")} of ${fr.full_shape.join("×")} · ${fr.lossy || ""}`; }
// points3d frame -> line segments / points in box coordinates of `box`
const wrapped = (a, b, box) => a.some((v, k) => Math.abs(v - b[k]) > 0.5 * box.s[k]);
function pointsGeometry(fr, box) {
  const { q } = decoded(fr), [n, m] = fr.shape, d = fr.dim, E = fr.extent, seg = [], heads = [];
  const P = k => { const r = [0, 0, 0]; for (let a = 0; a < 3; a++) {
    const phys = a < d ? E[a][0] + (E[a][1] - E[a][0]) * q[k * d + a] / 65534 : (box.ext[a][0] + box.ext[a][1]) / 2;
    r[a] = toBox(box, a, phys); } return r; };
  for (let l = 0; l < n; l++) { let prev = null;
    for (let j = 0; j < m; j++) { const k = l * m + j;
      if (q[k * d] === 65535) { prev = null; continue; }
      const p = P(k); if (prev && m > 1 && !wrapped(prev, p, box)) seg.push(...prev, ...p); prev = p; }
    if (prev) heads.push(...prev); }
  return { seg: new Float32Array(seg), heads: new Float32Array(heads) };
}
V3.update = function (R, force) {
  V3.R = R;
  const panel = document.getElementById("view3dPanel"); if (!panel) return;
  renderMixing(R);
  const frs = R ? Object.values(R.frames || {}) : [];
  const f3 = frs.filter(f => f.kind === "frame3d"), vec = frs.filter(f => f.kind === "vec2d");
  const pts = frs.filter(f => f.kind === "points3d" && f.dim === 3), pts2 = frs.filter(f => f.kind === "points3d" && f.dim === 2);
  panel.hidden = !(f3.length || pts.length || vec.length || pts2.length);
  if (panel.hidden) return;
  if (!built) build(panel);
  if (runKey !== R.id) { runKey = R.id; last = {}; }
  const three = f3.length || pts.length;                 // 3D block only with 3D content (2D runs: 2D block)
  document.getElementById("v3three").hidden = !three;
  renderLic(R, vec, pts2, frs.filter(f => !f.kind || f.kind === "frame"));
  if (!three) return;
  const eul = f3.filter(f => f.scale !== "dye").map(f => f.name).sort(), dyes = f3.filter(f => f.scale === "dye").map(f => f.name).sort();
  ctrl.field = options(document.getElementById("v3field"), eul, ctrl.field);
  ctrl.dye = options(document.getElementById("v3dye"), ["", ...dyes], ctrl.dye);
  const F = R.frames[ctrl.field], D = ctrl.dye ? R.frames[ctrl.dye] : null;
  if (F && !ctrl.signUser) {                   // declared defaults: vortex cores are lambda2 < 0, Q > 0
    ctrl.sign = F.scale !== "signed" ? "±" : /lambda|λ/i.test(F.name) ? "-" : /^q$|^q_|q.?crit/i.test(F.name) ? "+" : "±";
    const sel = document.getElementById("v3sign"); if (sel) sel.value = ctrl.sign; }
  const stamp = f => f ? f.name + "@" + f.wall + "#" + f.iteration : "";
  const [ev, lv] = views;
  if (!ev.gl) { fallback2d(F); }
  else {
    const clipKey = stamp(F) + "|" + ctrl.clipPct + "|" + ctrl.sign;
    if (F && (force || last.F !== clipKey)) {
      ev.clip = { ...clipScale(F, decoded(F).q, ctrl.clipPct, ctrl.sign), sign: ctrl.sign };
      document.getElementById("v3clipv").textContent = F.scale === "signed"
        ? `|v| ≤ ${fmtv(ev.clip.A)} (p${ctrl.clipPct})` : `[${fmtv(ev.clip.lo)}, ${fmtv(ev.clip.hi)}] (p${(100 - ctrl.clipPct).toFixed(1)}–p${ctrl.clipPct})`;
      const t0 = performance.now(); ev.setVolume(F, decoded(F).q, F.scale); ev.axes = F.axes; last.F = clipKey; last.iso = null;
      V3.stats.upload_ms = +(performance.now() - t0).toFixed(2); }
    if (!F) ev.clearVolume();
    // isosurface(s): recomputed only when the frame or the level changes
    const isoKey = clipKey + "|" + ctrl.isoF + "|" + ctrl.iso;
    if (F && last.iso !== isoKey) {
      ev.layers = ev.layers.filter(l => l.kind !== "mesh"); last.iso = isoKey;
      document.getElementById("v3isov").textContent = "";
      if (ctrl.iso) {
        const t0 = performance.now(), q = decoded(F).q, lv8 = x => 255 * (x - F.min) / ((F.max - F.min) || 1);
        const levels = [];
        // level = fraction isoF of the clipped scale; signed: the selected sign(s) only
        if (F.scale === "signed") { const A = ev.clip.A * ctrl.isoF;
          if (ctrl.sign !== "-") levels.push([A, GLASS]); if (ctrl.sign !== "+") levels.push([-A, RUBRIC]); }
        else levels.push([ev.clip.lo + ctrl.isoF * (ev.clip.hi - ev.clip.lo), SILVER]);
        let tri = 0;
        for (const [v, col] of levels) { const q8 = lv8(v); if (q8 <= 0 || q8 >= 255) continue;
          const m = isosurface(q, F.shape, q8, F.coords, ev.box); tri += m.n / 3;
          ev.layers.push({ kind: "mesh", key: "iso" + v, pos: m.pos, nor: m.nor, n: m.n, color: col }); }
        V3.stats.iso_ms = +(performance.now() - t0).toFixed(1); V3.stats.iso_triangles = tri;
        document.getElementById("v3isov").textContent = levels.map(l => (l[0] >= 0 && F.scale === "signed" ? "+" : "") + fmtv(l[0])).join(" / ")
          + ` · ${tri} tri · ${V3.stats.iso_ms} ms`;
      }
    }
    // Lagrangian view: dye volume (fluorescein) + points; box = Eulerian box if any, else the points' extent
    const baseExt = F ? F.extent : (D ? D.extent : (pts[0] && pts[0].dim === 3 ? pts[0].extent : [[0, 1], [0, 1], [0, 1]]));
    const lkey = stamp(D) + "|" + pts.map(stamp).join("|") + "|" + JSON.stringify(baseExt);
    if (force || last.L !== lkey) {
      last.L = lkey;
      if (D) { lv.setVolume(D, decoded(D).q, "dye"); } else { lv.clearVolume(); lv.box = boxOf(baseExt); }
      lv.axes = (F || D || {}).axes; lv.layers = [];
      pts.sort((a, b) => a.name < b.name ? -1 : 1).forEach((p, k) => {
        const g = pointsGeometry(p, lv.box), tracked = !D && k === 0;      // one dye colour per view
        const col = tracked ? DYE : SILVER;
        if (g.seg.length) lv.layers.push({ kind: "lines", key: "l" + k, pos: g.seg, color: col, size: 1 });
        if (g.heads.length) lv.layers.push({ kind: "points", key: "h" + k, pos: g.heads, color: p.mode === "particles" ? col : SILVER,
          size: p.mode === "particles" ? (ctrl.mobile ? 2.5 : 3) : 2 });
      });
    }
    const lcfg = { ...ctrl, slices: null, volume: !!D };
    lv.ctrl = lcfg;
    redraw();
  }
  document.getElementById("v3ecap").textContent = F
    ? `Eulerian · ${F.name} (${F.scale}) · t=${fmtv(F.t ?? "–")} · [${fmtv(F.min)}, ${fmtv(F.max)}] · ${lossy(F)}`
      + (ev.clip ? ` · display: ${F.scale === "signed" ? `sign ${ctrl.sign}, |v| clipped at the ${ctrl.clipPct}th percentile = ${fmtv(ev.clip.A)}` : `clipped to the ${(100 - ctrl.clipPct).toFixed(1)}–${ctrl.clipPct}th percentiles`} of the received volume (client side)` : "") : "Eulerian · no 3D field frame";
  document.getElementById("v3lcap").textContent = "Lagrangian · " + (pts.map(p =>
    `${p.name} (${p.mode}) ${p.shape.join("×")} of ${p.full_shape.join("×")}, stride ${p.stride.join("/")}, 16-bit` +
    (p.clipped ? `, ${p.clipped} clipped` : "")).join(" · ") || "no particle frame") + (D ? ` · dye ${D.name}: ${lossy(D)}` : "")
    + (pts.some(p => p.shape[1] > 1) ? " · segments crossing a periodic boundary not drawn" : "");
};
// 2D mid-plane fallback (no WebGL2): three canvases of the selected field
function fallback2d(F) {
  const note = document.getElementById("v3note"); if (!F) return;
  let box = document.getElementById("v3fb");
  if (!box) { box = el(`<div id="v3fb" class="frames"></div>`); note.after(box); }
  const q = decoded(F).q, [n0, n1, n2] = F.shape; box.innerHTML = "";
  for (const a of [0, 1, 2]) {
    const [w, h] = a === 0 ? [n2, n1] : a === 1 ? [n0, n2] : [n0, n1], m = [n0, n1, n2][a] >> 1;
    const img = new ImageData(w, h), A = Math.max(Math.abs(F.min), Math.abs(F.max)) || 1;
    for (let x = 0; x < w; x++) for (let y = 0; y < h; y++) {
      const [i, j, k] = a === 0 ? [m, y, x] : a === 1 ? [x, m, y] : [x, y, m], v8 = q[(i * n1 + j) * n2 + k];
      const v = F.min + (F.max - F.min) * v8 / 255, c = F.scale === "signed" ? scaleColor("signed", 0.5 + 0.5 * v / A) : scaleColor(F.scale === "dye" ? "dye" : "smoke", v8 / 255);
      img.data.set([c[0], c[1], c[2], 255], 4 * ((h - 1 - y) * w + x)); }
    const cv = document.createElement("canvas"); cv.width = w; cv.height = h; cv.style.width = "100%"; cv.style.imageRendering = "auto";
    cv.getContext("2d").putImageData(img, 0, 0);
    const d = el(`<div class="chart"><div class="t"><span>${esc3(F.name)} mid-plane ${"xyz"[a]}</span></div></div>`); d.appendChild(cv); box.appendChild(d); }
}
// 2D Lagrangian panel: particles / lines (fluorescein, the followed quantity) over a chosen 2D
// frame (smoke, or signed if it changes sign) or lab black; periodic wraps are not drawn as segments
function renderLag2(R, vec, pts2, f2d) {
  const fig = document.getElementById("v3lag2").parentElement; fig.hidden = !pts2.length; if (!pts2.length) return;
  const bgName = options(document.getElementById("v3bg"), ["", ...f2d.map(f => f.name).sort()],
    last.bgInit ? undefined : (f2d.find(f => /dye/.test(f.name)) || {}).name || "");
  last.bgInit = true;
  const bg = bgName ? R.frames[bgName] : null, E = (vec[0] || pts2[0]).extent;
  const key = pts2.map(p => p.name + p.wall + "#" + p.iteration).join("|") + "|" + (bg ? bg.name + bg.wall : "");
  if (last.lag2 === key) return; last.lag2 = key;
  const cv = document.getElementById("v3lag2"), Wc = cv.clientWidth || 600, asp = (E[1][1] - E[1][0]) / (E[0][1] - E[0][0]);
  const W = Math.min(ctrl.mobile ? 360 : 640, Math.round(Wc)), H = Math.max(40, Math.round(W * asp));
  cv.width = W; cv.height = H; cv.style.height = Math.round(Wc * asp) + "px";
  const g = cv.getContext("2d"); g.fillStyle = "#0B0B0B"; g.fillRect(0, 0, W, H);
  if (bg) {                                           // 2D frame: x = axis 0 horizontal, axis 1 up
    const [nx, ny] = bg.shape, raw = atob(bg.data), img = new ImageData(nx, ny), A = Math.max(Math.abs(bg.min), Math.abs(bg.max)) || 1;
    const signed = bg.min < 0 && bg.max > 0;
    for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) {
      const q = raw.charCodeAt(i * ny + j), v = bg.min + (bg.max - bg.min) * q / 255;
      const c = signed ? scaleColor("signed", 0.5 + 0.5 * v / A) : scaleColor("smoke", q / 255);
      img.data.set([c[0], c[1], c[2], 255], 4 * ((ny - 1 - j) * nx + i)); }
    const off = document.createElement("canvas"); off.width = nx; off.height = ny; off.getContext("2d").putImageData(img, 0, 0);
    g.imageSmoothingEnabled = true; g.drawImage(off, 0, 0, W, H);
  }
  for (const p of pts2) {
    const { q } = decoded(p), [n, m] = p.shape, pe = p.extent;
    // thin, semi-transparent fluorescein: thousands of tracers and long streaklines stay legible
    g.fillStyle = "rgba(198,240,60,.7)"; g.strokeStyle = "rgba(198,240,60,.45)"; g.lineWidth = 0.6;
    const X = k => ((pe[0][0] + (pe[0][1] - pe[0][0]) * q[2 * k] / 65534) - E[0][0]) / (E[0][1] - E[0][0]) * W;
    const Y = k => H - ((pe[1][0] + (pe[1][1] - pe[1][0]) * q[2 * k + 1] / 65534) - E[1][0]) / (E[1][1] - E[1][0]) * H;
    for (let l = 0; l < n; l++) { g.beginPath(); let started = false, px = 0, py = 0;
      for (let j = 0; j < m; j++) { const k = l * m + j; if (q[2 * k] === 65535) { started = false; continue; }
        const x = X(k), y = Y(k);
        if (m === 1) { g.fillRect(x - 0.6, y - 0.6, 1.2, 1.2); continue; }
        if (!started || Math.abs(x - px) > W / 2 || Math.abs(y - py) > H / 2) g.moveTo(x, y); else g.lineTo(x, y);
        started = true; px = x; py = y; }
      if (m > 1) g.stroke(); }
  }
  document.getElementById("v3lag2cap").textContent = "Lagrangian 2D · " + pts2.map(p =>
    `${p.name} (${p.mode}) ${p.shape.join("×")} of ${p.full_shape.join("×")}, 16-bit` + (p.clipped ? `, ${p.clipped} clipped` : "")).join(" · ")
    + (bg ? ` · under: ${bg.name} [${fmtv(bg.min)}, ${fmtv(bg.max)}] ${bg.shape.join("×")} of ${bg.full_shape.join("×")}, 8-bit` : "")
    + (pts2.some(p => p.shape[1] > 1) ? " · segments crossing a periodic boundary not drawn" : "");
}
function renderLic(R, vec, pts2, f2d) {
  const wrap = document.getElementById("v3two"); wrap.hidden = !(vec.length || pts2.length);
  if (wrap.hidden) return;
  renderLag2(R, vec, pts2, f2d);
  document.getElementById("v3lic").parentElement.hidden = !vec.length;
  if (!vec.length) return;
  const name = options(document.getElementById("v3vec"), vec.map(v => v.name).sort()), fr = R.frames[name];
  const cv = document.getElementById("v3lic"), Wc = cv.clientWidth || 600;
  const [nx, ny] = fr.shape, E = fr.extent, asp = (E[1][1] - E[1][0]) / (E[0][1] - E[0][0]);
  const W = Math.min(ctrl.mobile ? 320 : 400, Math.round(Wc)), H = Math.max(40, Math.round(W * asp));
  const stream = document.getElementById("v3st").checked, key = name + "@" + fr.wall + "#" + fr.iteration + "|" + W + "|" + stream;
  if (last.lic === key) return;
  // LIC runs on the main thread (~0.4-1.4 s measured in software): a new *frame* is re-convolved at
  // most every 5 s; a change made by the user (field, streamlines, size) is applied at once
  const userChange = !last.lic || last.lic.split("|").slice(1).join("|") !== key.split("|").slice(1).join("|") || !last.lic.startsWith(name + "@");
  if (!userChange && performance.now() - (last.licT || 0) < 5000) { if (!last.licPending) { last.licPending = true; setTimeout(() => { last.licPending = false; V3.update(V3.R); }, 5100); } return; }
  last.lic = key; last.licT = performance.now();
  const t0 = performance.now(), L = lic(fr, decoded(fr).q, W, H, Math.max(6, W / 40));
  cv.width = W; cv.height = H; cv.style.height = Math.round(Wc * asp) + "px";
  const g = cv.getContext("2d"); g.putImageData(L.img, 0, 0);
  if (stream) {                                 // streamlines from a regular seed grid (RK2 in pixels)
    g.strokeStyle = "rgba(237,235,230,.35)"; g.lineWidth = 0.8;
    for (let sx = 0.5; sx < 8; sx++) for (let sy = 0.5; sy < 8 * asp; sy++) {
      let x = sx * W / 8, y = sy * W / 8; if (y >= H) continue; g.beginPath(); g.moveTo(x, y);
      for (let k = 0; k < 160; k++) { const [u, v] = L.vel(x, y), m = Math.hypot(u, v); if (m < 0.02 * Math.min(W, H) / 8) break;
        x += 1.5 * u / m; y += 1.5 * v / m; if (x < 0 || y < 0 || x >= W || y >= H) break; g.lineTo(x, y); }
      g.stroke(); }
  }
  V3.stats.lic_ms = +(performance.now() - t0).toFixed(1);
  document.getElementById("v3liccap").textContent = `${name} · t=${fmtv(fr.t ?? "–")} · |u| ≤ ${fmtv(fr.amp)} · ${lossy(fr)} · LIC ${W}×${H} px, ${V3.stats.lic_ms} ms (grey = LIC, brightness = |u| on the smoke scale)`;
}
// ------------------------------------------------------------------------------ mixing curves
// scalars named mix_* (variance, mix-norms, APE/BPE, efficiency) grouped in their own panel
function renderMixing(R) {
  const panel = document.getElementById("mixPanel"), box = document.getElementById("mixCharts");
  if (!panel || typeof lineChart !== "function") return;
  const keys = new Set(); (R ? R.scalars : []).forEach(r => Object.keys(r).forEach(k => { if (k.startsWith("mix_")) keys.add(k); }));
  panel.hidden = !keys.size; if (!keys.size) return;
  box.innerHTML = "";
  const xiter = document.getElementById("xiter") && document.getElementById("xiter").checked;
  const rows = R.scalars.filter(r => Object.keys(r).some(k => k.startsWith("mix_")));
  const xs = rows.map(r => xiter ? r.iteration : r.t);
  for (const k of [...keys].sort()) {
    const ys = rows.map(r => r[k]), pos = ys.filter(y => typeof y === "number" && y > 0);
    const logy = pos.length > 1 && pos.length === ys.filter(y => typeof y === "number").length && Math.max(...pos) / Math.min(...pos) > 1e2;
    const d = el(`<div class="chart"><div class="t"><span>${esc3(k.slice(4))}${logy ? " (log)" : ""}</span><span>${esc3(fmtv(ys[ys.length - 1]))}</span></div><canvas></canvas></div>`);
    box.appendChild(d); lineChart(d.querySelector("canvas"), xs, ys, { logy });
  }
}
V3._test = { isosurface, lic, scaleColor, physToTex, boxOf };
})();
