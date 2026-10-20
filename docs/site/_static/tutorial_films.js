// Tutorial films (docs/site/tutorial_media.py): muted looping videos of the tutorials' own runs.
// They declare preload="none", so nothing is downloaded until one nears the viewport; a film then
// plays only while visible and not paused by the reader (one Pause / Play button per film, external
// review 0.1.0, finding 75), and never under prefers-reduced-motion (its poster stays, no button),
// as the films of the website (web/src/components/Film.tsx).
(function () {
  var films = document.querySelectorAll("video.cn-film");
  if (!films.length || typeof IntersectionObserver === "undefined") return;
  var motion = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
  var reduced = function () { return !!(motion && motion.matches); };
  var visible = new Map(), paused = new Map(), buttons = new Map();
  var label = function (v) {
    var b = buttons.get(v);
    if (!b) return;
    b.hidden = reduced();
    // the name says the action and contains the visible word (a voice command "click Pause"
    // works); no pressed state, which with a changing label read as "Play, pressed"
    b.textContent = paused.get(v) ? "Play" : "Pause";
    b.setAttribute("aria-label", (paused.get(v) ? "Play" : "Pause") + " the film");
  };
  var sync = function (v) {
    label(v);
    if (reduced() || paused.get(v) || !visible.get(v)) { v.pause(); return; }
    var p = v.play();
    if (p && p.catch) p.catch(function () { /* autoplay refused: the poster stays */ });
  };
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) { visible.set(e.target, e.isIntersecting); sync(e.target); });
  }, { rootMargin: "200px 0px" });
  films.forEach(function (v) {
    v.muted = true;
    var b = document.createElement("button");
    b.type = "button";
    b.className = "cn-film-btn";
    b.addEventListener("click", function () { paused.set(v, !paused.get(v)); sync(v); });
    buttons.set(v, b);
    v.insertAdjacentElement("afterend", b);
    io.observe(v);
    label(v);
  });
  if (motion && motion.addEventListener) {
    motion.addEventListener("change", function () { films.forEach(sync); });
  }
})();
