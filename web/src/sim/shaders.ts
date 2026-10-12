// GLSL ES 3.00 sources for the Kelvin-Helmholtz hero solver (kh.ts).
//
// Units: the domain is [0, Lx] x [0, 1] (Ly = 1), periodic in x, free-slip walls at y = 0 and 1.
// Every field lives on a cell-centred texture addressed by uv in [0,1]^2; a physical velocity
// (u, v) moves a point by dt * (u / Lx, v) in uv. The velocity/pressure grid is Nx x Ny with
// square cells (Nx = Lx * Ny); the two passive dyes (R = smoke, G = fluorescein) live on a grid
// 2-3x finer, so the smoke stays thinner than the flow resolution.

/** Full-screen triangle, no vertex buffer (gl_VertexID only). */
export const VERT = `#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const HEAD = `#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;
`;

/** Base state u = tanh(y / delta) plus a streamfunction perturbation (fundamental + subharmonic). */
export const INIT_VEL = HEAD + `
uniform vec2 uRes;      // grid size (Nx, Ny)
uniform float uLx;      // domain length (Ly = 1)
uniform float uDelta;   // shear-layer half-thickness
uniform float uEps;     // perturbation amplitude (fraction of U)
uniform vec4 uMode;     // (k1, phase1, k2, phase2), physical wavenumbers
uniform float uSub;     // relative amplitude of the subharmonic
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  float x = uv.x * uLx;
  float eta = (uv.y - 0.5) / uDelta;
  float g = exp(-0.5 * eta * eta);
  float c = cos(uMode.x * x + uMode.y) + uSub * cos(uMode.z * x + uMode.w);
  float s = uMode.x * sin(uMode.x * x + uMode.y) + uSub * uMode.z * sin(uMode.z * x + uMode.w);
  // psi = eps * delta * g(eta) * c(x):  u' = d(psi)/dy,  v' = -d(psi)/dx  (divergence-free)
  float up = -uEps * eta * g * c;
  float vp = uEps * uDelta * g * s;
  o = vec4(tanh(eta) + up, vp, 0.0, 0.0);
}`;

/** Dye initial condition: fine horizontal smoke lines (brighter near the layer) + one fluorescein filament. */
export const INIT_DYE = HEAD + `
uniform vec2 uRes;      // dye grid size
uniform float uLines;   // number of smoke lines across Ly
uniform float uDyeY;    // height of the fluorescein filament
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  float px = uRes.y / uLines;                                  // line spacing in texels
  float d = abs(fract(uv.y * uLines) - 0.5) * px;              // texels to the nearest line
  float env = 0.035 + 0.765 * exp(-pow((uv.y - 0.5) / 0.17, 2.0));
  float smoke = env * exp(-0.8 * d * d);
  float dd = (uv.y - uDyeY) * uRes.y;
  float dye = exp(-dd * dd / 2.5);
  o = vec4(smoke * (1.0 - dye), dye, 0.0, 0.0);
}`;

/**
 * Semi-Lagrangian step hat(x) = phi(x - dt u(x)): the whole velocity update (plus the explicit
 * viscous term nu dt lap u^n), and the MacCormack predictor for the dyes (uNuDt = 0).
 */
export const ADVECT = HEAD + `
uniform sampler2D uVel, uSrc;
uniform vec2 uRes;      // resolution of the target (= of uSrc)
uniform vec2 uScale;    // (1/Lx, 1): physical velocity -> uv velocity
uniform float uDt;
uniform float uNuDt;    // nu * dt / h^2
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 d = uDt * texture(uVel, uv).xy * uScale;
  vec4 r = texture(uSrc, uv - d);
  if (uNuDt > 0.0) {
    vec2 t = 1.0 / uRes;
    r += uNuDt * (texture(uSrc, uv + vec2(t.x, 0.0)) + texture(uSrc, uv - vec2(t.x, 0.0))
                + texture(uSrc, uv + vec2(0.0, t.y)) + texture(uSrc, uv - vec2(0.0, t.y))
                - 4.0 * texture(uSrc, uv));
  }
  o = r;
}`;

/**
 * MacCormack corrector with a min/max limiter (Selle et al. 2008), fused with the backward step:
 * phi^{n+1} = hat + (phi - hat(x + dt u)) / 2, clamped to the 4 texels around the departure point.
 * Used for the dyes only: on the colocated velocity grid the non-dissipative corrector feeds the
 * odd-even pressure mode (dark crosses in the vortex cores), so the velocity stays semi-Lagrangian.
 */
export const CORRECT = HEAD + `
uniform sampler2D uVel, uSrc, uHat;
uniform vec2 uRes;
uniform vec2 uScale;
uniform float uDt;
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 t = 1.0 / uRes;
  vec2 d = uDt * texture(uVel, uv).xy * uScale;
  vec4 phi = texture(uSrc, uv);
  vec4 hat = texture(uHat, uv);
  vec4 r = hat + 0.5 * (phi - texture(uHat, uv + d));
  // limiter: extrema of the stencil used by the predictor's bilinear lookup
  vec2 c = (floor((uv - d) * uRes - 0.5) + 0.5) * t;
  vec4 a = texture(uSrc, c), b = texture(uSrc, c + vec2(t.x, 0.0));
  vec4 e = texture(uSrc, c + vec2(0.0, t.y)), f = texture(uSrc, c + t);
  o = clamp(r, min(min(a, b), min(e, f)), max(max(a, b), max(e, f)));
}`;

// Pressure projection on the colocated grid (grid units, h = 1). x wraps; at the walls the normal
// velocity is reflected (v_ghost = -v) and the pressure has a zero normal gradient (p_ghost = p).
const FETCH = `
uniform ivec2 uN;
ivec2 cellE(ivec2 c) { return ivec2((c.x + 1) % uN.x, c.y); }
ivec2 cellW(ivec2 c) { return ivec2((c.x + uN.x - 1) % uN.x, c.y); }
ivec2 cellN(ivec2 c) { return ivec2(c.x, min(c.y + 1, uN.y - 1)); }
ivec2 cellS(ivec2 c) { return ivec2(c.x, max(c.y - 1, 0)); }
`;

/** div u (central differences). */
export const DIVERGENCE = HEAD + FETCH + `
uniform sampler2D uVel;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  float vC = texelFetch(uVel, c, 0).y;
  float vN = c.y == uN.y - 1 ? -vC : texelFetch(uVel, cellN(c), 0).y;
  float vS = c.y == 0 ? -vC : texelFetch(uVel, cellS(c), 0).y;
  float du = texelFetch(uVel, cellE(c), 0).x - texelFetch(uVel, cellW(c), 0).x;
  o = vec4(0.5 * (du + vN - vS), 0.0, 0.0, 0.0);
}`;

/** One Jacobi sweep for lap p = div u (5-point, warm-started from the previous frame's p). */
export const JACOBI = HEAD + FETCH + `
uniform sampler2D uP, uDiv;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  float s = texelFetch(uP, cellE(c), 0).x + texelFetch(uP, cellW(c), 0).x
          + texelFetch(uP, cellN(c), 0).x + texelFetch(uP, cellS(c), 0).x;
  o = vec4(0.25 * (s - texelFetch(uDiv, c, 0).x), 0.0, 0.0, 0.0);
}`;

/** u <- u - grad p; no-penetration imposed on the wall rows. */
export const GRADIENT = HEAD + FETCH + `
uniform sampler2D uP, uVel;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  vec2 u = texelFetch(uVel, c, 0).xy;
  u.x -= 0.5 * (texelFetch(uP, cellE(c), 0).x - texelFetch(uP, cellW(c), 0).x);
  u.y -= 0.5 * (texelFetch(uP, cellN(c), 0).x - texelFetch(uP, cellS(c), 0).x);
  if (c.y == 0 || c.y == uN.y - 1) u.y = 0.0;
  o = vec4(u, 0.0, 0.0);
}`;

/** Pointer forcing: a Gaussian momentum impulse (x-periodic distance). */
export const SPLAT = HEAD + `
uniform sampler2D uVel;
uniform vec2 uRes;
uniform vec2 uAt;       // uv of the pointer
uniform vec2 uForce;    // physical velocity increment at the centre
uniform float uLx;
uniform float uRadius;  // in Ly units
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 d = uv - uAt;
  d.x = (d.x - floor(d.x + 0.5)) * uLx;
  o = texture(uVel, uv) + vec4(uForce * exp(-dot(d, d) / (uRadius * uRadius)), 0.0, 0.0);
}`;

/**
 * Dye injection by the reader: fluorescein (G) laid along the pointer segment A -> B travelled since
 * the last step, as a Gaussian tube of radius uRadius (x-periodic distance, Ly units), clamped to 1.
 * A passive scalar source: it does not change the velocity (the momentum impulse is SPLAT).
 */
export const SPLAT_DYE = HEAD + `
uniform sampler2D uSrc;
uniform vec2 uRes;
uniform vec2 uA, uB;    // uv of the segment ends (B may be unwrapped past 1 in x)
uniform float uLx;
uniform float uRadius;
void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  vec2 ab = (uB - uA) * vec2(uLx, 1.0);
  vec2 ap = uv - uA;
  ap.x = (ap.x - floor(ap.x + 0.5)) * uLx;
  float h = clamp(dot(ap, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
  vec2 d = ap - h * ab;
  vec4 s = texture(uSrc, uv);
  float g = exp(-dot(d, d) / (uRadius * uRadius));
  o = vec4(s.x * (1.0 - g), min(1.0, s.y + g), 0.0, 0.0);
}`;

/**
 * Display: silver-gelatin smoke (#EDEBE6 on #0B0B0B), one fluorescein filament (#C6F03C) with a
 * faint halo, animated film grain and a slight vignette. The view is a window of height uSpan
 * (in Ly) centred on the layer, scrolled in x by uScroll (the layer drifts downstream).
 */
export const DISPLAY = HEAD + `
uniform sampler2D uDye, uGhost;
uniform float uGhostMix;  // cross-fade weight of the previous cycle's dyes
uniform vec2 uView;       // canvas size in pixels
uniform float uSpan, uScroll, uLx, uSeed;
uniform vec2 uDyeTexel;
float hash(vec2 p) {
  p = fract(p * vec2(443.897, 441.423));
  p += dot(p, p.yx + 19.19);
  return fract((p.x + p.y) * p.x);
}
vec2 dyeAt(vec2 uv) {
  vec2 d = texture(uDye, uv).xy;
  return uGhostMix > 0.0 ? mix(d, texture(uGhost, uv).xy, uGhostMix) : d;
}
void main() {
  vec2 fc = gl_FragCoord.xy;
  vec2 uv = vec2(uScroll + (fc.x - 0.5 * uView.x) / uView.y * uSpan / uLx,
                 0.5 + (fc.y / uView.y - 0.5) * uSpan);
  vec2 d = dyeAt(uv);
  // halo of the dye: 6-tap ring, ~2.5 dye texels
  float g = 0.0;
  for (int i = 0; i < 6; i++) {
    float a = float(i) * 1.0472;
    g += dyeAt(uv + 3.0 * uDyeTexel * vec2(cos(a), sin(a))).y;
  }
  vec3 lab = vec3(0.0431), silver = vec3(0.929, 0.922, 0.902), dye = vec3(0.776, 0.941, 0.235);
  float s = 1.0 - exp(-2.0 * d.x);                    // soft shoulder: dense smoke saturates
  vec3 col = mix(lab, silver, 0.92 * s);
  col = mix(col, dye, smoothstep(0.1, 0.6, d.y));   // crisp filament, diffuse tails fade out
  col += dye * 0.05 * g;
  vec2 q = fc / uView - 0.5;
  col *= 1.0 - 0.28 * smoothstep(0.25, 0.75, length(q * vec2(0.9, 1.25)));
  float n = hash(fc + uSeed) + hash(fc.yx * 1.37 - uSeed) - 1.0;   // triangular grain
  col += n * (0.018 + 0.045 * s);
  o = vec4(col, 1.0);
}`;
