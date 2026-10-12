import{a as e,c as t,r as n,s as r,t as i}from"./index-BGAuUEVG.js";import{n as a,t as o}from"./ui-DeGmfevc.js";import{i as s,r as c,t as l}from"./Executed-DzsYRYX2.js";var u=t(r(),1),d=`#version 300 es
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`,f=`#version 300 es
precision highp float;
precision highp sampler2D;
out vec4 o;
`,p=f+`
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
}`,m=f+`
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
}`,h=f+`
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
}`,ee=f+`
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
}`,g=`
uniform ivec2 uN;
ivec2 cellE(ivec2 c) { return ivec2((c.x + 1) % uN.x, c.y); }
ivec2 cellW(ivec2 c) { return ivec2((c.x + uN.x - 1) % uN.x, c.y); }
ivec2 cellN(ivec2 c) { return ivec2(c.x, min(c.y + 1, uN.y - 1)); }
ivec2 cellS(ivec2 c) { return ivec2(c.x, max(c.y - 1, 0)); }
`,_=f+g+`
uniform sampler2D uVel;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  float vC = texelFetch(uVel, c, 0).y;
  float vN = c.y == uN.y - 1 ? -vC : texelFetch(uVel, cellN(c), 0).y;
  float vS = c.y == 0 ? -vC : texelFetch(uVel, cellS(c), 0).y;
  float du = texelFetch(uVel, cellE(c), 0).x - texelFetch(uVel, cellW(c), 0).x;
  o = vec4(0.5 * (du + vN - vS), 0.0, 0.0, 0.0);
}`,te=f+g+`
uniform sampler2D uP, uDiv;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  float s = texelFetch(uP, cellE(c), 0).x + texelFetch(uP, cellW(c), 0).x
          + texelFetch(uP, cellN(c), 0).x + texelFetch(uP, cellS(c), 0).x;
  o = vec4(0.25 * (s - texelFetch(uDiv, c, 0).x), 0.0, 0.0, 0.0);
}`,v=f+g+`
uniform sampler2D uP, uVel;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  vec2 u = texelFetch(uVel, c, 0).xy;
  u.x -= 0.5 * (texelFetch(uP, cellE(c), 0).x - texelFetch(uP, cellW(c), 0).x);
  u.y -= 0.5 * (texelFetch(uP, cellN(c), 0).x - texelFetch(uP, cellS(c), 0).x);
  if (c.y == 0 || c.y == uN.y - 1) u.y = 0.0;
  o = vec4(u, 0.0, 0.0);
}`,y=f+`
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
}`,ne=f+`
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
}`,re=f+`
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
}`,b=1/28,x=1/28e3,S=.002,ie=24,C=.62,ae=.012,oe=4;function se(e,t,n){let r=e.createShader(e.FRAGMENT_SHADER);e.shaderSource(r,n),e.compileShader(r);let i=e.createProgram();return e.attachShader(i,t),e.attachShader(i,r),e.linkProgram(i),{p:i,fs:r}}function ce(e,{p:t,fs:n}){if(!e.getProgramParameter(t,e.LINK_STATUS)){let r=e.getShaderParameter(n,e.COMPILE_STATUS)?`link: `+e.getProgramInfoLog(t):`shader: `+e.getShaderInfoLog(n);throw e.deleteShader(n),Error(r)}e.deleteShader(n);let r={},i=e.getProgramParameter(t,e.ACTIVE_UNIFORMS);for(let n=0;n<i;n++){let i=e.getActiveUniform(t,n).name;r[i]=e.getUniformLocation(t,i)}return{p:t,u:r}}var le=typeof requestAnimationFrame==`function`?e=>requestAnimationFrame(e):e=>setTimeout(()=>e(performance.now()),16),ue=typeof cancelAnimationFrame==`function`?e=>cancelAnimationFrame(e):e=>clearTimeout(e),de=()=>new Promise(e=>le(()=>e())),fe=()=>new Promise(e=>setTimeout(e,0));function pe(e){return()=>{e=e+1831565813|0;let t=Math.imul(e^e>>>15,1|e);return t=t+Math.imul(t^t>>>7,61|t)^t,((t^t>>>14)>>>0)/4294967296}}function me(){return typeof matchMedia==`function`&&matchMedia(`(max-width: 640px)`).matches||(navigator.hardwareConcurrency||8)<=4}async function he(e,t={},n=()=>!1){let r=e.getContext(`webgl2`,{alpha:!1,antialias:!1,depth:!1,stencil:!1,premultipliedAlpha:!1,preserveDrawingBuffer:!1,powerPreference:`high-performance`,failIfMajorPerformanceCaveat:!t.allowSoftware});if(!r)throw Error(`WebGL2 unavailable`);if(!t.allowSoftware){let e=r.getExtension(`WEBGL_debug_renderer_info`),t=String(r.getParameter(e?e.UNMASKED_RENDERER_WEBGL:r.RENDERER));if(/SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(t))throw Error(`software renderer: `+t)}let i=r.getExtension(`EXT_color_buffer_float`),a=r.getExtension(`OES_texture_float_linear`),o=r.getExtension(`EXT_color_buffer_half_float`);if(!i&&!o)throw Error(`no float render targets`);let s=!!(i&&a),c=t.weak??me(),l=typeof HTMLCanvasElement<`u`&&e instanceof HTMLCanvasElement?e:null,u=t.view??(l?{width:()=>l.clientWidth,height:()=>l.clientHeight,dpr:()=>window.devicePixelRatio||1}:(()=>{throw Error(`an OffscreenCanvas needs opts.view`)})()),f=t.ny??(c?128:256),g=t.dyeScale??(c?2:3),he=t.maxDpr??(c?1:1.5),ge=t.cycle??32,_e=t.interactive??!0,ve=t.adaptive??!0,w=r.createShader(r.VERTEX_SHADER);r.shaderSource(w,d),r.compileShader(w);let T={initVel:p,initDye:m,advect:h,correct:ee,div:_,jacobi:te,grad:v,splat:y,splatDye:ne,show:re},E=Object.fromEntries(Object.entries(T).map(([e,t])=>[e,se(r,w,t)])),D=()=>{for(let e in E)r.deleteProgram(E[e].p),r.deleteShader(E[e].fs);r.deleteShader(w)},O=r.getExtension(`KHR_parallel_shader_compile`);if(O)for(let e=0;e<600;e++){if(n())throw D(),Error(`cancelled`);if(Object.values(E).every(e=>r.getProgramParameter(e.p,O.COMPLETION_STATUS_KHR)))break;await de()}let k={};for(let e of Object.keys(T)){if(O||await fe(),n())throw D(),Error(`cancelled`);k[e]=ce(r,E[e])}let A=r.createVertexArray();r.bindVertexArray(A),r.disable(r.BLEND),r.disable(r.DEPTH_TEST);let j=[];function M(e,t,n,i,a){let o=r.createTexture();r.bindTexture(r.TEXTURE_2D,o);let s=n===`rg`?i?r.RG32F:r.RG16F:i?r.R32F:r.R16F;r.texImage2D(r.TEXTURE_2D,0,s,e,t,0,n===`rg`?r.RG:r.RED,i?r.FLOAT:r.HALF_FLOAT,null);let c=a?r.LINEAR:r.NEAREST;r.texParameteri(r.TEXTURE_2D,r.TEXTURE_MIN_FILTER,c),r.texParameteri(r.TEXTURE_2D,r.TEXTURE_MAG_FILTER,c),r.texParameteri(r.TEXTURE_2D,r.TEXTURE_WRAP_S,r.REPEAT),r.texParameteri(r.TEXTURE_2D,r.TEXTURE_WRAP_T,r.CLAMP_TO_EDGE);let l=r.createFramebuffer();if(r.bindFramebuffer(r.FRAMEBUFFER,l),r.framebufferTexture2D(r.FRAMEBUFFER,r.COLOR_ATTACHMENT0,r.TEXTURE_2D,o,0),r.checkFramebufferStatus(r.FRAMEBUFFER)!==r.FRAMEBUFFER_COMPLETE)throw Error(`incomplete framebuffer`);let u={tex:o,fbo:l,w:e,h:t};return j.push(u),u}function N(e,t,n,r,i){let a={read:M(e,t,n,r,i),write:M(e,t,n,r,i),swap(){let e=a.read;a.read=a.write,a.write=e}};return a}let P=2,F=512,I,L,R,z,B,ye;function be(){for(let e of j)r.deleteTexture(e.tex),r.deleteFramebuffer(e.fbo);j.length=0}function xe(){be(),F=Math.round(P*f);let e=e=>{I=N(F,f,`rg`,e,!0),L=N(F,f,`r`,e,!1),R=M(F,f,`r`,e,!1),z=N(g*F,g*f,`rg`,!1,!0),B=N(g*F,g*f,`rg`,!1,!0),ye=M(g*F,g*f,`rg`,!1,!0)};try{e(s)}catch(t){if(!s)throw t;be(),s=!1,e(!1)}}function Se(){let e=u.width()/Math.max(1,u.height());return Math.min(4,Math.max(2,Math.ceil(e*C*2+.25)/2))}function V(t){r.bindFramebuffer(r.FRAMEBUFFER,t?t.fbo:null),r.viewport(0,0,t?t.w:e.width,t?t.h:e.height)}function H(e,t){r.useProgram(e.p);let n=0;for(let i in t)r.activeTexture(r.TEXTURE0+n),r.bindTexture(r.TEXTURE_2D,t[i]),r.uniform1i(e.u[i],n++)}let U=()=>r.drawArrays(r.TRIANGLES,0,3),Ce=7;function we(){let e=pe(Ce++),t=Math.round(P*2),n=2*Math.PI*t/P;H(k.initVel,{}),r.uniform2f(k.initVel.u.uRes,F,f),r.uniform1f(k.initVel.u.uLx,P),r.uniform1f(k.initVel.u.uDelta,b),r.uniform1f(k.initVel.u.uEps,.02),r.uniform4f(k.initVel.u.uMode,n,2*Math.PI*e(),n/2,2*Math.PI*e()),r.uniform1f(k.initVel.u.uSub,.08+.08*e()),V(I.read),U(),V(L.read),r.clearColor(0,0,0,0),r.clear(r.COLOR_BUFFER_BIT),H(k.initDye,{}),r.uniform2f(k.initDye.u.uRes,g*F,g*f),r.uniform1f(k.initDye.u.uLines,Math.round(g*f/12)),r.uniform1f(k.initDye.u.uDyeY,.5535714285714286),V(z.read),U()}function W(e,t,n){let i=e.read;if(H(k.advect,{uVel:I.read.tex,uSrc:i.tex}),r.uniform2f(k.advect.u.uRes,i.w,i.h),r.uniform2f(k.advect.u.uScale,1/P,1),r.uniform1f(k.advect.u.uDt,S),r.uniform1f(k.advect.u.uNuDt,n),V(t??e.write),U(),!t){e.swap();return}H(k.correct,{uVel:I.read.tex,uSrc:i.tex,uHat:t.tex}),r.uniform2f(k.correct.u.uRes,i.w,i.h),r.uniform2f(k.correct.u.uScale,1/P,1),r.uniform1f(k.correct.u.uDt,S),V(e.write),U(),e.swap()}function Te(){H(k.div,{uVel:I.read.tex}),r.uniform2i(k.div.u.uN,F,f),V(R),U(),H(k.jacobi,{}),r.uniform2i(k.jacobi.u.uN,F,f);for(let e=0;e<ie;e++)H(k.jacobi,{uP:L.read.tex,uDiv:R.tex}),V(L.write),U(),L.swap();H(k.grad,{uP:L.read.tex,uVel:I.read.tex}),r.uniform2i(k.grad.u.uN,F,f),V(I.write),U(),I.swap()}let G=null,K=[0,0],q=null,J=null;function Ee(){G&&(K[0]!==0||K[1]!==0)&&(H(k.splat,{uVel:I.read.tex}),r.uniform2f(k.splat.u.uRes,F,f),r.uniform2f(k.splat.u.uAt,G[0],G[1]),r.uniform2f(k.splat.u.uForce,K[0],K[1]),r.uniform1f(k.splat.u.uLx,P),r.uniform1f(k.splat.u.uRadius,.035),V(I.write),U(),I.swap(),K=[0,0]),q&&J&&(H(k.splatDye,{uSrc:z.read.tex}),r.uniform2f(k.splatDye.u.uRes,g*F,g*f),r.uniform2f(k.splatDye.u.uA,q[0],q[1]),r.uniform2f(k.splatDye.u.uB,J[0],J[1]),r.uniform1f(k.splatDye.u.uLx,P),r.uniform1f(k.splatDye.u.uRadius,2.2/(g*f)),V(z.write),U(),z.swap(),q=J)}let De=0,Oe=0,Y=0,ke=0;function Ae(){if(Ee(),W(I,null,x*S*f*f),Te(),W(z,ye,0),Y>0&&W(B,null,0),De+=1/60,Oe++,ke+=ae/60,Y>0&&(Y=Math.max(0,Y-1/60)),De>ge){let e=B;B=z,z=e,we(),De=0,Oe=0,Y=oe}}let je=0;function Me(){Pe(),H(k.show,Y>0?{uDye:z.read.tex,uGhost:B.read.tex}:{uDye:z.read.tex,uGhost:z.read.tex});let t=k.show.u;r.uniform1f(t.uGhostMix,Y>0?(Y/oe)**1.5:0),r.uniform2f(t.uView,e.width,e.height),r.uniform1f(t.uSpan,C),r.uniform1f(t.uScroll,ke),r.uniform1f(t.uLx,P),r.uniform1f(t.uSeed,je++%97*7.31),r.uniform2f(t.uDyeTexel,1/(g*F),1/(g*f)),V(null),U()}let Ne=1;function Pe(){let t=Math.min(u.dpr(),he)*Ne,n=Math.max(1,Math.round(u.width()*t)),r=Math.max(1,Math.round(u.height()*t));(e.width!==n||e.height!==r)&&(e.width=n,e.height=r)}function Fe(){P=Se(),xe(),we(),De=0,Oe=0,Y=0}let X=0,Ie=0,Z=0,Q=!1,Le=!1,Re=0,ze=0,Be=0;function Ve(e){X=le(Ve);let n=Ie?Math.min(100,e-Ie):1e3/60;Ie=e,Z+=n;let r=0;for(;Z>=1e3/60-1&&r<2;)Ae(),Z-=1e3/60,r++;if(Z>1e3/60&&(Z=0),Me(),Q||(Q=!0,t.onReady?.()),ve){if(Be<1e3){Be+=n;return}if(Re+=n,ze++,Re>=2e3){let e=1e3*ze/Re;Re=0,ze=0,e<40&&f>96&&(f=Math.max(96,f>>1),Ne*=.75,Be=0,Fe())}}}let He=!1,$=null;function Ue(e){let{x:t,y:n,w:r,h:i}=e;return t<0||n<0||t>r||n>i||!i?null:[ke+(t-.5*r)/i*C/P,.5+(.5-n/i)*C]}function We(e){if(e.button!==0)return;let t=Ue(e);if(!t)return;He=!0,$=[e.x,e.y];let n=[t[0]-Math.floor(t[0]),t[1]];q=n,J=[n[0]+1e-4,n[1]],G=n}function Ge(e){if(!He)return;let t=Ue(e);if(!t)return;if($){let t=.02/e.h*60;K=[Math.max(-.6,Math.min(.6,K[0]+(e.x-$[0])*t)),Math.max(-.6,Math.min(.6,K[1]-(e.y-$[1])*t))]}$=[e.x,e.y];let n=q?q[0]:0,r=t[0]-Math.floor(t[0]);J=[r+Math.round(n-r),t[1]],q||(q=[r,t[1]]),G=[r,t[1]]}function Ke(){He=!1,$=null,G=null,q=null,J=null}if(await fe(),n()){for(let e in k)r.deleteProgram(k[e].p);throw r.deleteShader(w),Error(`cancelled`)}function qe(e){e.preventDefault(),ue(X),X=0,t.onFail?.(`context lost`)}e.addEventListener(`webglcontextlost`,qe);let Je=0;function Ye(e,n){clearTimeout(Je);let r=e,i=8,a=()=>{let e=performance.now(),o=Math.min(i,r);for(let e=0;e<o;e++)Ae();r-=o;let s=performance.now()-e;i=Math.max(1,Math.min(120,Math.round(i*(s>0?25/s:2)))),r>0?Je=setTimeout(a,0):(Me(),Q||(Q=!0,t.onReady?.()),n?.())};Je=setTimeout(a,0)}return Pe(),Fe(),{start(){!X&&!Le&&(clearTimeout(Je),Ie=0,Z=0,X=le(Ve))},stop(){ue(X),X=0},pointer(e){_e&&(e.kind===`down`?We(e):e.kind===`move`?Ge(e):Ke())},still:Ye,step(e){for(let t=0;t<e;t++)Ae()},stats(){return{t:Oe*S/b,nx:F,ny:f,dyeScale:g,re:Math.round(b/x)}},render(){Me(),Q||(Q=!0,t.onReady?.())},resize(){Pe(),Math.abs(Se()-P)>.01&&Fe()},dispose(){Le=!0,ue(X),X=0,clearTimeout(Je),e.removeEventListener(`webglcontextlost`,qe),be();for(let e in k)r.deleteProgram(k[e].p);r.deleteShader(w),r.deleteVertexArray(A)}}}var ge=4e3;async function _e(e,t,n){if(typeof Worker==`function`&&typeof e.transferControlToOffscreen==`function`&&typeof location<`u`&&location.protocol!==`file:`){let r=await ve(e,t,n);if(r)return r}if(n())throw Error(`cancelled`);return he(e,t,n)}async function ve(e,t,n){let r;try{r=new Worker(new URL(`/conatus/assets/kh.worker-HKauHS69.js`,``+import.meta.url),{type:`module`})}catch{return null}let i=e=>new Promise(t=>{let n=setTimeout(()=>o(null),e),i=e=>o(e.data),a=()=>o(null);function o(e){clearTimeout(n),r.removeEventListener(`message`,i),r.removeEventListener(`error`,a),t(e)}r.addEventListener(`message`,i),r.addEventListener(`error`,a)}),a=i(ge);r.postMessage({type:`probe`,allowSoftware:!!t.allowSoftware});let o=await a;if(!o||o.probe===`none`||n()){if(r.terminate(),n())throw Error(`cancelled`);return null}if(o.probe===`software`)throw r.terminate(),Error(`software renderer`);let s=e.transferControlToOffscreen(),{onReady:c,onFail:l,...u}=t,d=i(3e4);r.postMessage({type:`init`,canvas:s,w:e.clientWidth,h:e.clientHeight,dpr:window.devicePixelRatio||1,opts:u},[s]);let f=await d;if(f!==`created`)throw r.terminate(),Error(typeof f==`object`&&f?f.error:`worker did not start`);let p;r.addEventListener(`message`,e=>{e.data===`ready`?c?.():e.data===`stilled`?p?.():e.data&&typeof e.data==`object`&&`fail`in e.data&&l?.(String(e.data.fail))}),r.addEventListener(`error`,()=>l?.(`worker error`));let m=e=>r.postMessage(e),h={t:0,nx:0,ny:0,dyeScale:0,re:0};return{start:()=>m({type:`start`}),stop:()=>m({type:`stop`}),step:()=>{},render:()=>m({type:`render`}),resize:()=>m({type:`resize`,w:e.clientWidth,h:e.clientHeight,dpr:window.devicePixelRatio||1}),pointer:e=>m({type:`pointer`,e}),still:(e,t)=>{p=t,m({type:`still`,n:e})},dispose:()=>{m({type:`dispose`}),setTimeout(()=>r.terminate(),1e3)},stats:()=>h}}var w=n(),T=1200,E=600,D=E/17.4,O=-Math.PI,k=T/D/(2*Math.PI),A=28/64,j=.05,M=.55,N=1.5,P=[];for(let e=0;(e+.5)*A<9.7;e++){let t=(e+.5)*A,n=.035+.765*Math.exp(-((t/28/.17)**2));for(let e of[1,-1])e===1&&Math.abs(t-N)<A/2||P.push({level:Math.cosh(t),sign:e,op:Math.min(.9,.12+.95*n),w:.7+.8*n})}var F=Math.cosh(N),I=e=>Math.round(e*10)/10,L=e=>I((e-O)*D),R=e=>I(E/2-D*e),z=64,B=8,ye=16,be=e=>e-j>=1?ye:z,xe=new Map;function Se(e){let t=xe.get(e);if(!t){let n=Math.round(k*e),r=O+T/D,i=new Float64Array(n+1),a=new Float64Array(n+1),o=[];for(let e=0;e<=n;e++)i[e]=O+(r-O)*e/n,a[e]=Math.cos(i[e]),o.push(String(L(i[e])));t={n,x:i,cos:a,sx:o},xe.set(e,t)}return t}function V(e,t,n,r=z){let i=O+T/D;if(e-n>=1){let i=Se(r),a=Array(i.n+1);for(let r=0;r<=i.n;r++)a[r]=(r?`L`:`M`)+i.sx[r]+` `+R(t*Math.acosh(Math.max(1,e+n*i.cos[r])));return a.join(``)}if(t===-1||e+n<=1)return``;let a=Math.acos(Math.max(-1,Math.min(1,(1-e)/n))),o=Math.max(12,Math.round(48*r/z)),s=``;for(let t=Math.floor(O/(2*Math.PI))-1;2*Math.PI*t-a<i;t++){let r=2*Math.PI*t;if(r+a<O)continue;let i=``,c=``;for(let t=0;t<=o;t++){let s=r-a+2*a*t/o,l=Math.acosh(Math.max(1,e+n*Math.cos(s)));i+=(i?`L`:`M`)+L(s)+` `+R(l),c=`L`+L(s)+` `+R(-l)+c}s+=i+c+`Z`}return s}function H({paths:e,dye:t}){return(0,w.jsxs)(`svg`,{viewBox:`0 0 ${T} ${E}`,preserveAspectRatio:`xMidYMid slice`,"aria-hidden":`true`,style:{position:`absolute`,inset:0,width:`100%`,height:`100%`},children:[(0,w.jsx)(`g`,{fill:`none`,stroke:`#EDEBE6`,strokeLinecap:`round`,strokeLinejoin:`round`,children:P.map((t,n)=>(0,w.jsx)(`path`,{ref:t=>{e.current[n]=t},d:V(t.level,t.sign,j,be(t.level)),strokeOpacity:t.op,strokeWidth:t.w},n))}),(0,w.jsx)(`path`,{ref:t,d:V(F,1,j,be(F)),fill:`none`,stroke:`#C6F03C`,strokeWidth:1.8,strokeOpacity:.95})]})}var U=`.cn-fluid-ui{position:absolute;top:12px;right:12px;z-index:2;display:flex;flex-direction:column;align-items:flex-end;gap:6px;
pointer-events:none;font:400 11px/1.3 var(--cn-font-code,monospace);color:#A3A29D;text-align:right}
.cn-fluid-row{display:flex;align-items:center;gap:8px}
.cn-fluid-hint{font-family:var(--cn-font-text,sans-serif);font-size:11px;color:#8A8A86;text-shadow:0 0 6px #0B0B0B}
.cn-fluid-btn{pointer-events:auto;display:inline-flex;align-items:center;gap:.45em;padding:.3em .6em;
font:400 11px/1 var(--cn-font-text,sans-serif);letter-spacing:.04em;
color:#A3A29D;background:transparent;border:1px solid rgba(237,235,230,.18);border-radius:var(--cn-radius,2px);cursor:pointer}
.cn-fluid-btn:hover{color:#EDEBE6;border-color:rgba(237,235,230,.5)}
.cn-fluid-btn:focus-visible{outline:2px solid #C6F03C;outline-offset:2px}
.cn-fluid-btn svg{width:.8em;height:.8em;fill:currentColor}`;function Ce(e){let{className:t,label:n,interactive:r=!0}=e,i=(0,u.useRef)(null),a=(0,u.useRef)(null),o=(0,u.useRef)(()=>{}),s=(0,u.useRef)(!1),c=(0,u.useRef)([]),l=(0,u.useRef)(null),[d,f]=(0,u.useState)(!1),[p,m]=(0,u.useState)(!1),[h,ee]=(0,u.useState)(!1),[g,_]=(0,u.useState)(!1);return(0,u.useEffect)(()=>{let e=i.current;if(!e)return;let t=()=>{},n=0,u=0,d=0,f=!1,p=0,h=()=>typeof matchMedia==`function`&&matchMedia(`(prefers-reduced-motion: reduce)`).matches,g=(e,t,n=0,r=P.length)=>{for(let i=n;i<r;i++)c.current[i]?.setAttribute(`d`,V(P[i].level,P[i].sign,e,t));r===P.length&&l.current?.setAttribute(`d`,V(F,1,e,t))},_=e=>{let t=0,n=()=>{if(f)return;let r=performance.now();for(;t<P.length&&performance.now()-r<6;)g(e,z,t,t+1),t++;t<P.length?d=window.setTimeout(n,0):g(e,z,P.length,P.length)};n()},te=()=>{if(f)return;if(h()){_(M);return}let e=performance.now(),t=16,n=-1e9,r=i=>{if(f)return;if(i-n<31){p=requestAnimationFrame(r);return}n=i;let a=Math.max(0,Math.min(1,(i-e)/4200)),o=a<.5?4*a*a*a:1-(-2*a+2)**3/2;if(a>=1){_(M);return}let s=performance.now();g(j+.5*o,t);let c=performance.now()-s;c>6&&t>B?t/=2:c<1.5&&t<z&&(t*=2),p=requestAnimationFrame(r)};p=requestAnimationFrame(r)},v=async()=>{if(f)return;let n,i=/[?&]sim=soft\b/.test(window.location.search+window.location.hash),c=matchMedia(`(max-width: 640px)`).matches||(navigator.hardwareConcurrency||8)<=4;try{n=await _e(e,{interactive:r,allowSoftware:i,weak:c,onReady:()=>m(!0),onFail:()=>{m(!1),ee(!1),te()}},()=>f)}catch{f||te();return}if(f){n.dispose();return}a.current=n;let l=matchMedia(`(prefers-reduced-motion: reduce)`),u=!0,d=l.matches,p=!1,h=!1,g=()=>{h||(h=!0,n.still(480,()=>{h=!1,p=!0}))},_=()=>{if(ee(!d),d){n.stop(),p||g();return}u&&!document.hidden&&!s.current?n.start():n.stop()};o.current=_;let v=new IntersectionObserver(([e])=>{u=e.isIntersecting&&e.intersectionRatio>=.35,_()},{threshold:[0,.35,.7,1]});v.observe(e);let y=()=>_();document.addEventListener(`visibilitychange`,y);let ne=()=>{d=l.matches,p=!1,_()};l.addEventListener(`change`,ne);let re=0,b=new ResizeObserver(()=>{clearTimeout(re),re=window.setTimeout(()=>{n.resize(),d?(p=!1,_()):n.render()},150)});b.observe(e);let x=t=>r=>{let i=e.getBoundingClientRect();if(t===`down`&&r.button===0)try{e.setPointerCapture(r.pointerId)}catch{}n.pointer({kind:t,x:r.clientX-i.left,y:r.clientY-i.top,w:i.width,h:i.height,button:r.button})},S=x(`down`),ie=x(`move`),C=x(`up`);r&&(e.addEventListener(`pointerdown`,S),e.addEventListener(`pointermove`,ie),e.addEventListener(`pointerup`,C),e.addEventListener(`pointercancel`,C)),_(),t=()=>{v.disconnect(),b.disconnect(),clearTimeout(re),e.removeEventListener(`pointerdown`,S),e.removeEventListener(`pointermove`,ie),e.removeEventListener(`pointerup`,C),e.removeEventListener(`pointercancel`,C),document.removeEventListener(`visibilitychange`,y),l.removeEventListener(`change`,ne),n.dispose(),a.current=null,o.current=()=>{}}},y=()=>{typeof window.requestIdleCallback==`function`?n=window.requestIdleCallback(()=>{v()},{timeout:2500}):u=setTimeout(()=>{v()},1200)};return document.readyState===`complete`?y():window.addEventListener(`load`,y,{once:!0}),()=>{f=!0,cancelAnimationFrame(p),window.removeEventListener(`load`,y),n&&window.cancelIdleCallback(n),clearTimeout(u),clearTimeout(d),t()}},[r]),(0,w.jsxs)(`div`,{className:t,role:`group`,"aria-label":n,style:{position:`relative`,width:`100%`,height:`100%`,overflow:`hidden`,pointerEvents:`none`},children:[(0,w.jsx)(`style`,{children:U}),(0,w.jsx)(`div`,{"aria-hidden":`true`,style:{position:`absolute`,inset:0,zIndex:-2,background:`#0B0B0B`,opacity:+!p,transition:`opacity .8s`},children:(0,w.jsx)(H,{paths:c,dye:l})}),(0,w.jsx)(`canvas`,{ref:i,"aria-hidden":`true`,style:{position:`absolute`,inset:0,zIndex:-2,width:`100%`,height:`100%`,display:`block`,opacity:+!!p,transition:`opacity .8s`,pointerEvents:r&&p?`auto`:`none`,touchAction:`pan-y`,cursor:r&&p?`crosshair`:void 0},onPointerDown:()=>_(!0)}),h&&p&&(0,w.jsxs)(`div`,{className:`cn-fluid-ui`,children:[(0,w.jsx)(`div`,{className:`cn-fluid-row`,children:(0,w.jsxs)(`button`,{type:`button`,className:`cn-fluid-btn`,"aria-pressed":d,onClick:()=>{s.current=!s.current,f(s.current),o.current()},children:[(0,w.jsx)(`svg`,{viewBox:`0 0 10 10`,"aria-hidden":`true`,children:d?(0,w.jsx)(`path`,{d:`M2 1l7 4-7 4z`}):(0,w.jsx)(`path`,{d:`M2 1h2v8H2zM6 1h2v8H6z`})}),d?`Play`:`Pause`]})}),r&&!g&&(0,w.jsx)(`span`,{className:`cn-fluid-hint`,children:`Drag across the layer to inject dye`})]})]})}var we=`web/src/pages/Home.tsx`,W=s(`t01_orr_sommerfeld`),Te=c(W.code,`problem`,we).replace(/\nprint\([^\n]*explain\(\)\)[^\n]*$/,``)+`

`+c(W.code,`solve`,we),G=[/^c = /,/^Re_c = /,/\[PASS\] (right residual|spectral tail)/,/^OK:/],K=[{k:`Write`,title:`Equations as on paper`,text:`Declare the domain, the fields and the equations in their usual form; name the boundaries and say what holds there. Conatus places the rest and tells you what it did.`},{k:`Simulate`,title:`On a CPU or a GPU`,text:`The same model runs on your laptop or on a GPU with one keyword. Resolution, scheme and time step are always yours to choose.`},{k:`Analyse`,title:`Where a flow becomes unstable`,text:`Eigenvalues with their residuals, critical parameters, exact adjoints, transient growth and Floquet analysis of time-periodic flows.`},{k:`Discover`,title:`Orbits and bifurcations`,text:`Equilibria, travelling waves and periodic orbits by Newton–Krylov; branches followed through folds, pitchforks and Hopf points.`},{k:`Keep track`,title:`Every run, recorded`,text:`Parameters, code version and results are kept for each run, so you can find it, compare it and reproduce it later.`}];function q(){return(0,w.jsxs)(w.Fragment,{children:[(0,w.jsxs)(`section`,{className:`hero`,"aria-labelledby":`hero-title`,children:[(0,w.jsx)(`div`,{className:`hero-sim`,children:(0,w.jsx)(Ce,{interactive:!0,label:`Live Kelvin–Helmholtz simulation`})}),(0,w.jsx)(`div`,{className:`hero-veil`,"aria-hidden":`true`}),(0,w.jsxs)(`div`,{className:`wrap hero-content`,children:[(0,w.jsx)(`p`,{className:`hero-eyebrow`,children:`Spectral PDEs · Dynamical systems · GPU`}),(0,w.jsxs)(`h1`,{id:`hero-title`,className:`hero-title`,children:[`Write the equations.`,(0,w.jsx)(`br`,{}),`Conatus finds the structure.`]}),(0,w.jsx)(`p`,{className:`hero-lede`,children:`A spectral framework for partial differential equations and the dynamical systems they define, native on GPUs. Write the equations as on paper, simulate them, find where the flow becomes unstable, follow its orbits and bifurcations.`}),(0,w.jsxs)(`div`,{className:`row hero-cta`,children:[(0,w.jsx)(o,{to:`/tutorials/getting-started`,children:`Get started`}),(0,w.jsx)(o,{to:`/documentation`,variant:`ghost`,children:`Documentation`})]}),(0,w.jsxs)(`p`,{className:`hero-status`,children:[`Version `,i.version,` · CPU supported, GPU in beta · `,i.repo?(0,w.jsx)(`a`,{href:i.repo,children:`source on GitHub`}):(0,w.jsx)(w.Fragment,{children:`repository private during the preview, source on request`}),` `,`· install: `,(0,w.jsx)(`code`,{children:`pip install conatus`}),` (`,(0,w.jsx)(e,{to:`/install`,children:`Install`}),`)`]})]}),(0,w.jsx)(`p`,{className:`hero-caption wrap`,children:`A Kelvin–Helmholtz shear layer rolling up, computed live in your browser.`})]}),(0,w.jsx)(`section`,{className:`section lab home-code`,"aria-labelledby":`code-h`,children:(0,w.jsxs)(`div`,{className:`wrap`,children:[(0,w.jsxs)(`div`,{className:`section-head`,children:[(0,w.jsx)(`p`,{className:`eyebrow`,children:`Write`}),(0,w.jsx)(`h2`,{id:`code-h`,children:`The equation on the page is the equation that runs`}),(0,w.jsx)(`p`,{className:`section-intro`,children:`Conatus reads the problem as mathematics, compiles it for the CPU or the GPU, and tells you exactly what it did. Nothing changes behind your back. Below, the first tutorial as it ran, and what it printed.`})]}),(0,w.jsxs)(`div`,{className:`code-grid`,children:[(0,w.jsx)(a,{title:`${W.code.path.split(`/`).pop()} (excerpt)`,children:Te}),(0,w.jsxs)(`div`,{className:`code-side`,children:[(0,w.jsx)(l,{out:W.out,patterns:G,page:we,label:`Output`}),(0,w.jsxs)(`ol`,{className:`code-steps`,children:[(0,w.jsxs)(`li`,{children:[(0,w.jsx)(`strong`,{children:`Declare`}),` the domain, the fields and the equations, as written on paper.`]}),(0,w.jsxs)(`li`,{children:[(0,w.jsx)(`strong`,{children:`Discretise and check.`}),` A rank report refuses a missing or extra boundary condition before anything runs.`]}),(0,w.jsxs)(`li`,{children:[(0,w.jsx)(`strong`,{children:`Solve, and see how far to trust it.`}),` Every result comes with its own checks.`]})]})]})]})]})}),(0,w.jsx)(`section`,{className:`section does`,"aria-labelledby":`does-h`,children:(0,w.jsxs)(`div`,{className:`wrap`,children:[(0,w.jsxs)(`div`,{className:`section-head`,children:[(0,w.jsx)(`p`,{className:`eyebrow`,children:`What it does`}),(0,w.jsx)(`h2`,{id:`does-h`,children:`From the equations to the structure of the flow`})]}),(0,w.jsx)(`ol`,{className:`does-grid`,children:K.map(e=>(0,w.jsxs)(`li`,{className:`does-item`,children:[(0,w.jsx)(`p`,{className:`does-k`,children:e.k}),(0,w.jsx)(`h3`,{children:e.title}),(0,w.jsx)(`p`,{children:e.text})]},e.k))}),(0,w.jsx)(`p`,{className:`more-link`,children:(0,w.jsxs)(e,{to:`/capabilities`,children:[`All capabilities `,(0,w.jsx)(`span`,{"aria-hidden":`true`,children:`→`})]})})]})}),(0,w.jsx)(`section`,{className:`section start`,"aria-labelledby":`start-h`,children:(0,w.jsxs)(`div`,{className:`wrap`,children:[(0,w.jsxs)(`div`,{className:`section-head`,children:[(0,w.jsx)(`p`,{className:`eyebrow`,children:`Get started`}),(0,w.jsx)(`h2`,{id:`start-h`,children:`Install, learn, read, cite`}),(0,w.jsx)(`p`,{className:`section-intro`,children:`Install it, work through the tutorials, then find each part described once in the documentation.`})]}),(0,w.jsxs)(`div`,{className:`start-grid`,children:[(0,w.jsxs)(e,{className:`start-card`,to:`/install`,children:[(0,w.jsx)(`span`,{className:`start-k`,children:`Install`}),(0,w.jsx)(`strong`,{children:(0,w.jsx)(`code`,{children:`pip install conatus`})}),(0,w.jsxs)(`span`,{children:[`CPU wheel with the compiled core, no compiler needed; a CUDA 12 wheel for NVIDIA GPUs (beta). `,(0,w.jsx)(`code`,{children:`python -m conatus doctor`}),` says what this machine can run.`]})]}),(0,w.jsxs)(e,{className:`start-card`,to:`/tutorials`,children:[(0,w.jsx)(`span`,{className:`start-k`,children:`Learn`}),(0,w.jsx)(`strong`,{children:`Tutorials`}),(0,w.jsx)(`span`,{children:`Executed tutorials, each ending on a published result: start with 0, getting started.`})]}),(0,w.jsxs)(e,{className:`start-card`,to:`/documentation`,children:[(0,w.jsx)(`span`,{className:`start-k`,children:`Read`}),(0,w.jsx)(`strong`,{children:`Documentation`}),(0,w.jsx)(`span`,{children:`The user guide, the API reference and the mathematics behind each method.`})]}),(0,w.jsxs)(e,{className:`start-card`,to:`/citing`,children:[(0,w.jsx)(`span`,{className:`start-k`,children:`Cite`}),(0,w.jsx)(`strong`,{children:`Citing`}),(0,w.jsx)(`span`,{children:`CITATION.cff and BibTeX. Licence: BSD-3-Clause.`})]}),(0,w.jsxs)(e,{className:`start-card`,to:`/coming-next`,children:[(0,w.jsx)(`span`,{className:`start-k`,children:`Next`}),(0,w.jsx)(`strong`,{children:`Coming next`}),(0,w.jsx)(`span`,{children:`Multi-GPU runs, the Studio, an interface for AI agents and further stability tools: under development, not in 0.1.0.`})]})]})]})})]})}export{q as default};