(function(){let e=1/28,t=.002,n=.62;function r(e,t,n){let r=e.createShader(e.FRAGMENT_SHADER);e.shaderSource(r,n),e.compileShader(r);let i=e.createProgram();return e.attachShader(i,t),e.attachShader(i,r),e.linkProgram(i),{p:i,fs:r}}function i(e,{p:t,fs:n}){if(!e.getProgramParameter(t,e.LINK_STATUS)){let r=e.getShaderParameter(n,e.COMPILE_STATUS)?`link: `+e.getProgramInfoLog(t):`shader: `+e.getShaderInfoLog(n);throw e.deleteShader(n),Error(r)}e.deleteShader(n);let r={},i=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let n=0;n<i;n++){let i=e.getActiveUniform(t,n).name;r[i]=e.getUniformLocation(t,i)}return{p:t,u:r}}let a=typeof requestAnimationFrame==`function`?e=>requestAnimationFrame(e):e=>setTimeout(()=>e(performance.now()),16),o=typeof cancelAnimationFrame==`function`?e=>cancelAnimationFrame(e):e=>clearTimeout(e),s=()=>new Promise(e=>a(()=>e())),c=()=>new Promise(e=>setTimeout(e,0));function ee(e){return()=>{e=e+1831565813|0;let t=Math.imul(e^e>>>15,1|e);return t=t+Math.imul(t^t>>>7,61|t)^t,((t^t>>>14)>>>0)/4294967296}}function te(){return typeof matchMedia==`function`&&matchMedia(`(max-width: 640px)`).matches||(navigator.hardwareConcurrency||8)<=4}async function l(l,u={},d=()=>!1){let f=l.getContext(`webgl2`,{alpha:!1,antialias:!1,depth:!1,stencil:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,powerPreference:`high-performance`,failIfMajorPerformanceCaveat:!u.allowSoftware});if(!f)throw Error(`WebGL2 unavailable`);if(!u.allowSoftware){let e=f.getExtension(`WEBGL_debug_renderer_info`),t=String(f.getParameter(e?e.UNMASKED_RENDERER_WEBGL:f.RENDERER));if(/SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(t))throw Error(`software renderer: `+t)}let p=f.getExtension(`EXT_color_buffer_float`),m=f.getExtension(`OES_texture_float_linear`),h=f.getExtension(`EXT_color_buffer_half_float`);if(!p&&!h)throw Error(`no float render targets`);let g=!!(p&&m),_=u.weak??te(),v=typeof HTMLCanvasElement<`u`&&l instanceof HTMLCanvasElement?l:null,y=u.view??(v?{width:()=>v.clientWidth,height:()=>v.clientHeight,dpr:()=>window.devicePixelRatio||1}:(()=>{throw Error(`an OffscreenCanvas needs opts.view`)})()),b=u.ny??(_?128:256),x=u.dyeScale??(_?2:3),ne=u.maxDpr??(_?1:1.5),re=u.cycle??32,ie=u.interactive??!0,ae=u.adaptive??!0,S=f.createShader(f.VERTEX_SHADER);f.shaderSource(S,`#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`),f.compileShader(S);let oe={initVel:`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;

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
}`,initDye:`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;

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
}`,advect:`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;

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
}`,correct:`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;

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
}`,div:`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;

uniform ivec2 uN;
ivec2 cellE(ivec2 c) { return ivec2((c.x + 1) % uN.x, c.y); }
ivec2 cellW(ivec2 c) { return ivec2((c.x + uN.x - 1) % uN.x, c.y); }
ivec2 cellN(ivec2 c) { return ivec2(c.x, min(c.y + 1, uN.y - 1)); }
ivec2 cellS(ivec2 c) { return ivec2(c.x, max(c.y - 1, 0)); }

uniform sampler2D uVel;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  float vC = texelFetch(uVel, c, 0).y;
  float vN = c.y == uN.y - 1 ? -vC : texelFetch(uVel, cellN(c), 0).y;
  float vS = c.y == 0 ? -vC : texelFetch(uVel, cellS(c), 0).y;
  float du = texelFetch(uVel, cellE(c), 0).x - texelFetch(uVel, cellW(c), 0).x;
  o = vec4(0.5 * (du + vN - vS), 0.0, 0.0, 0.0);
}`,jacobi:`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;

uniform ivec2 uN;
ivec2 cellE(ivec2 c) { return ivec2((c.x + 1) % uN.x, c.y); }
ivec2 cellW(ivec2 c) { return ivec2((c.x + uN.x - 1) % uN.x, c.y); }
ivec2 cellN(ivec2 c) { return ivec2(c.x, min(c.y + 1, uN.y - 1)); }
ivec2 cellS(ivec2 c) { return ivec2(c.x, max(c.y - 1, 0)); }

uniform sampler2D uP, uDiv;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  float s = texelFetch(uP, cellE(c), 0).x + texelFetch(uP, cellW(c), 0).x
          + texelFetch(uP, cellN(c), 0).x + texelFetch(uP, cellS(c), 0).x;
  o = vec4(0.25 * (s - texelFetch(uDiv, c, 0).x), 0.0, 0.0, 0.0);
}`,grad:`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;

uniform ivec2 uN;
ivec2 cellE(ivec2 c) { return ivec2((c.x + 1) % uN.x, c.y); }
ivec2 cellW(ivec2 c) { return ivec2((c.x + uN.x - 1) % uN.x, c.y); }
ivec2 cellN(ivec2 c) { return ivec2(c.x, min(c.y + 1, uN.y - 1)); }
ivec2 cellS(ivec2 c) { return ivec2(c.x, max(c.y - 1, 0)); }

uniform sampler2D uP, uVel;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  vec2 u = texelFetch(uVel, c, 0).xy;
  u.x -= 0.5 * (texelFetch(uP, cellE(c), 0).x - texelFetch(uP, cellW(c), 0).x);
  u.y -= 0.5 * (texelFetch(uP, cellN(c), 0).x - texelFetch(uP, cellS(c), 0).x);
  if (c.y == 0 || c.y == uN.y - 1) u.y = 0.0;
  o = vec4(u, 0.0, 0.0);
}`,splat:`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;

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
}`,splatDye:`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;

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
}`,show:`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;

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
}`},C=Object.fromEntries(Object.entries(oe).map(([e,t])=>[e,r(f,S,t)])),se=()=>{for(let e in C)f.deleteProgram(C[e].p),f.deleteShader(C[e].fs);f.deleteShader(S)},ce=f.getExtension(`KHR_parallel_shader_compile`);if(ce)for(let e=0;e<600;e++){if(d())throw se(),Error(`cancelled`);if(Object.values(C).every(e=>f.getProgramParameter(e.p,ce.COMPLETION_STATUS_KHR)))break;await s()}let w={};for(let e of Object.keys(oe)){if(ce||await c(),d())throw se(),Error(`cancelled`);w[e]=i(f,C[e])}let le=f.createVertexArray();f.bindVertexArray(le),f.disable(f.BLEND),f.disable(f.DEPTH_TEST);let T=[];function E(e,t,n,r,i){let a=f.createTexture();f.bindTexture(f.TEXTURE_2D,a);let o=n===`rg`?r?f.RG32F:f.RG16F:r?f.R32F:f.R16F;f.texImage2D(f.TEXTURE_2D,0,o,e,t,0,n===`rg`?f.RG:f.RED,r?f.FLOAT:f.HALF_FLOAT,null);let s=i?f.LINEAR:f.NEAREST;f.texParameteri(f.TEXTURE_2D,f.TEXTURE_MIN_FILTER,s),f.texParameteri(f.TEXTURE_2D,f.TEXTURE_MAG_FILTER,s),f.texParameteri(f.TEXTURE_2D,f.TEXTURE_WRAP_S,f.REPEAT),f.texParameteri(f.TEXTURE_2D,f.TEXTURE_WRAP_T,f.CLAMP_TO_EDGE);let c=f.createFramebuffer();if(f.bindFramebuffer(f.FRAMEBUFFER,c),f.framebufferTexture2D(f.FRAMEBUFFER,f.COLOR_ATTACHMENT0,f.TEXTURE_2D,a,0),f.checkFramebufferStatus(f.FRAMEBUFFER)!==f.FRAMEBUFFER_COMPLETE)throw Error(`incomplete framebuffer`);let ee={tex:a,fbo:c,w:e,h:t};return T.push(ee),ee}function D(e,t,n,r,i){let a={read:E(e,t,n,r,i),write:E(e,t,n,r,i),swap(){let e=a.read;a.read=a.write,a.write=e}};return a}let O=2,k=512,A,j,M,N,P,ue;function F(){for(let e of T)f.deleteTexture(e.tex),f.deleteFramebuffer(e.fbo);T.length=0}function de(){F(),k=Math.round(O*b);let e=e=>{A=D(k,b,`rg`,e,!0),j=D(k,b,`r`,e,!1),M=E(k,b,`r`,e,!1),N=D(x*k,x*b,`rg`,!1,!0),P=D(x*k,x*b,`rg`,!1,!0),ue=E(x*k,x*b,`rg`,!1,!0)};try{e(g)}catch(t){if(!g)throw t;F(),g=!1,e(!1)}}function fe(){let e=y.width()/Math.max(1,y.height());return Math.min(4,Math.max(2,Math.ceil(e*n*2+.25)/2))}function I(e){f.bindFramebuffer(f.FRAMEBUFFER,e?e.fbo:null),f.viewport(0,0,e?e.w:l.width,e?e.h:l.height)}function L(e,t){f.useProgram(e.p);let n=0;for(let r in t)f.activeTexture(f.TEXTURE0+n),f.bindTexture(f.TEXTURE_2D,t[r]),f.uniform1i(e.u[r],n++)}let R=()=>f.drawArrays(f.TRIANGLES,0,3),pe=7;function me(){let t=ee(pe++),n=Math.round(O*2),r=2*Math.PI*n/O;L(w.initVel,{}),f.uniform2f(w.initVel.u.uRes,k,b),f.uniform1f(w.initVel.u.uLx,O),f.uniform1f(w.initVel.u.uDelta,e),f.uniform1f(w.initVel.u.uEps,.02),f.uniform4f(w.initVel.u.uMode,r,2*Math.PI*t(),r/2,2*Math.PI*t()),f.uniform1f(w.initVel.u.uSub,.08+.08*t()),I(A.read),R(),I(j.read),f.clearColor(0,0,0,0),f.clear(f.COLOR_BUFFER_BIT),L(w.initDye,{}),f.uniform2f(w.initDye.u.uRes,x*k,x*b),f.uniform1f(w.initDye.u.uLines,Math.round(x*b/12)),f.uniform1f(w.initDye.u.uDyeY,.5535714285714286),I(N.read),R()}function z(e,n,r){let i=e.read;if(L(w.advect,{uVel:A.read.tex,uSrc:i.tex}),f.uniform2f(w.advect.u.uRes,i.w,i.h),f.uniform2f(w.advect.u.uScale,1/O,1),f.uniform1f(w.advect.u.uDt,t),f.uniform1f(w.advect.u.uNuDt,r),I(n??e.write),R(),!n){e.swap();return}L(w.correct,{uVel:A.read.tex,uSrc:i.tex,uHat:n.tex}),f.uniform2f(w.correct.u.uRes,i.w,i.h),f.uniform2f(w.correct.u.uScale,1/O,1),f.uniform1f(w.correct.u.uDt,t),I(e.write),R(),e.swap()}function he(){L(w.div,{uVel:A.read.tex}),f.uniform2i(w.div.u.uN,k,b),I(M),R(),L(w.jacobi,{}),f.uniform2i(w.jacobi.u.uN,k,b);for(let e=0;e<24;e++)L(w.jacobi,{uP:j.read.tex,uDiv:M.tex}),I(j.write),R(),j.swap();L(w.grad,{uP:j.read.tex,uVel:A.read.tex}),f.uniform2i(w.grad.u.uN,k,b),I(A.write),R(),A.swap()}let B=null,V=[0,0],H=null,U=null;function ge(){B&&(V[0]!==0||V[1]!==0)&&(L(w.splat,{uVel:A.read.tex}),f.uniform2f(w.splat.u.uRes,k,b),f.uniform2f(w.splat.u.uAt,B[0],B[1]),f.uniform2f(w.splat.u.uForce,V[0],V[1]),f.uniform1f(w.splat.u.uLx,O),f.uniform1f(w.splat.u.uRadius,.035),I(A.write),R(),A.swap(),V=[0,0]),H&&U&&(L(w.splatDye,{uSrc:N.read.tex}),f.uniform2f(w.splatDye.u.uRes,x*k,x*b),f.uniform2f(w.splatDye.u.uA,H[0],H[1]),f.uniform2f(w.splatDye.u.uB,U[0],U[1]),f.uniform1f(w.splatDye.u.uLx,O),f.uniform1f(w.splatDye.u.uRadius,2.2/(x*b)),I(N.write),R(),N.swap(),H=U)}let W=0,G=0,K=0,_e=0;function ve(){if(ge(),z(A,null,3571428571428572e-20*t*b*b),he(),z(N,ue,0),K>0&&z(P,null,0),W+=1/60,G++,_e+=.012/60,K>0&&(K=Math.max(0,K-1/60)),W>re){let e=P;P=N,N=e,me(),W=0,G=0,K=4}}let ye=0;function be(){Se(),L(w.show,K>0?{uDye:N.read.tex,uGhost:P.read.tex}:{uDye:N.read.tex,uGhost:N.read.tex});let e=w.show.u;f.uniform1f(e.uGhostMix,K>0?(K/4)**1.5:0),f.uniform2f(e.uView,l.width,l.height),f.uniform1f(e.uSpan,n),f.uniform1f(e.uScroll,_e),f.uniform1f(e.uLx,O),f.uniform1f(e.uSeed,ye++%97*7.31),f.uniform2f(e.uDyeTexel,1/(x*k),1/(x*b)),I(null),R()}let xe=1;function Se(){let e=Math.min(y.dpr(),ne)*xe,t=Math.max(1,Math.round(y.width()*e)),n=Math.max(1,Math.round(y.height()*e));(l.width!==t||l.height!==n)&&(l.width=t,l.height=n)}function Ce(){O=fe(),de(),me(),W=0,G=0,K=0}let q=0,J=0,Y=0,X=!1,we=!1,Z=0,Te=0,Ee=0;function De(e){q=a(De);let t=J?Math.min(100,e-J):1e3/60;J=e,Y+=t;let n=0;for(;Y>=1e3/60-1&&n<2;)ve(),Y-=1e3/60,n++;if(Y>1e3/60&&(Y=0),be(),X||(X=!0,u.onReady?.()),ae){if(Ee<1e3){Ee+=t;return}if(Z+=t,Te++,Z>=2e3){let e=1e3*Te/Z;Z=0,Te=0,e<40&&b>96&&(b=Math.max(96,b>>1),xe*=.75,Ee=0,Ce())}}}let Oe=!1,Q=null;function ke(e){let{x:t,y:r,w:i,h:a}=e;return t<0||r<0||t>i||r>a||!a?null:[_e+(t-.5*i)/a*n/O,.5+(.5-r/a)*n]}function Ae(e){if(e.button!==0)return;let t=ke(e);if(!t)return;Oe=!0,Q=[e.x,e.y];let n=[t[0]-Math.floor(t[0]),t[1]];H=n,U=[n[0]+1e-4,n[1]],B=n}function je(e){if(!Oe)return;let t=ke(e);if(!t)return;if(Q){let t=.02/e.h*60;V=[Math.max(-.6,Math.min(.6,V[0]+(e.x-Q[0])*t)),Math.max(-.6,Math.min(.6,V[1]-(e.y-Q[1])*t))]}Q=[e.x,e.y];let n=H?H[0]:0,r=t[0]-Math.floor(t[0]);U=[r+Math.round(n-r),t[1]],H||(H=[r,t[1]]),B=[r,t[1]]}function Me(){Oe=!1,Q=null,B=null,H=null,U=null}if(await c(),d()){for(let e in w)f.deleteProgram(w[e].p);throw f.deleteShader(S),Error(`cancelled`)}function Ne(e){e.preventDefault(),o(q),q=0,u.onFail?.(`context lost`)}l.addEventListener(`webglcontextlost`,Ne);let $=0;function Pe(e,t){clearTimeout($);let n=e,r=8,i=()=>{let e=performance.now(),a=Math.min(r,n);for(let e=0;e<a;e++)ve();n-=a;let o=performance.now()-e;r=Math.max(1,Math.min(120,Math.round(r*(o>0?25/o:2)))),n>0?$=setTimeout(i,0):(be(),X||(X=!0,u.onReady?.()),t?.())};$=setTimeout(i,0)}return Se(),Ce(),{start(){!q&&!we&&(clearTimeout($),J=0,Y=0,q=a(De))},stop(){o(q),q=0},pointer(e){ie&&(e.kind===`down`?Ae(e):e.kind===`move`?je(e):Me())},still:Pe,step(e){for(let t=0;t<e;t++)ve()},stats(){return{t:G*t/e,nx:k,ny:b,dyeScale:x,re:1e3}},render(){be(),X||(X=!0,u.onReady?.())},resize(){Se(),Math.abs(fe()-O)>.01&&Ce()},dispose(){we=!0,o(q),q=0,clearTimeout($),l.removeEventListener(`webglcontextlost`,Ne),F();for(let e in w)f.deleteProgram(w[e].p);f.deleteShader(S),f.deleteVertexArray(le)}}}let u=e=>self.postMessage(e),d=null,f=1,p=1,m=1,h=!1;function g(e){if(typeof OffscreenCanvas>`u`)return{probe:`none`,renderer:``};let t=new OffscreenCanvas(1,1).getContext(`webgl2`,{failIfMajorPerformanceCaveat:!e});if(!t)return{probe:e?`none`:`software`,renderer:``};let n=t.getExtension(`WEBGL_debug_renderer_info`),r=String(t.getParameter(n?n.UNMASKED_RENDERER_WEBGL:t.RENDERER));return t.getExtension(`WEBGL_lose_context`)?.loseContext(),{probe:/SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(r)&&!e?`software`:`ok`,renderer:r}}self.onmessage=async e=>{let t=e.data;switch(t.type){case`probe`:u(g(t.allowSoftware));break;case`init`:f=t.w,p=t.h,m=t.dpr;try{let e=await l(t.canvas,{...t.opts,view:{width:()=>f,height:()=>p,dpr:()=>m},onReady:()=>u(`ready`),onFail:e=>u({fail:e})},()=>h);if(h){e.dispose();break}d=e,u(`created`)}catch(e){u({error:String(e?.message??e)})}break;case`start`:d?.start();break;case`stop`:d?.stop();break;case`render`:d?.render();break;case`resize`:f=t.w,p=t.h,m=t.dpr,d?.resize();break;case`pointer`:d?.pointer(t.e);break;case`still`:d?.still(t.n,()=>u(`stilled`));break;case`dispose`:h=!0,d?.dispose(),d=null,self.close()}}})();