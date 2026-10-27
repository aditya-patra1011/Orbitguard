// OrbitGuard dashboard: the latest screening run on a 3-D globe.
//
// Data comes from the OrbitGuard API on the same origin:
//   /runs/latest, /runs/{id}/elements, /conjunctions, /pipeline/runs
// Every screened object is propagated in a Web Worker with SGP4 (satellite.js) from the
// exact element sets the run used, so the globe shows what screening saw.
"use strict";

const COLORS = { starlink: "#5aa9ff", oneweb: "#62d394", debris: "#ff8d4d", other: "#c8ccd4" };
const LABELS = { starlink: "Starlink", oneweb: "OneWeb", debris: "Debris", other: "Other" };
const ALERT_PC = 1e-4;          // the usual action threshold
const ALERT_KEY = "1e-4";       // the same threshold as the filter writes it
const UPDATE_MS = 250;          // how often the catalogue is re-propagated
const TRACK_MIN = 50;           // minutes of orbit drawn either side of TCA

const state = {
  run: null, objects: [], index: new Map(), selected: null, selectedEntities: [], encounterPoint: null,
};

// ------------------------------------------------------------------ small helpers

class HttpError extends Error {
  constructor(status, detail) { super(detail); this.status = status; }
}

async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) { /* not JSON */ }
    throw new HttpError(r.status, detail);
  }
  return r.json();
}

// ------------------------------------------------------------------ data source
// The same questions answered by the live API, or by the JSON files `orbitguard
// export-site` writes for GitHub Pages (config.js sets ORBITGUARD_DATA = "static").
// Thresholds are the strings used by the filter: "1e-4", "1e-6" or "0".

const STATIC = window.ORBITGUARD_DATA === "static";

const apiSource = {
  latestRun: () => getJSON("/runs/latest"),
  latestPipeline: () => getJSON("/pipeline/runs?limit=1").then((runs) => runs[0]),
  count: (run, minPc) => getJSON(`/conjunctions?run_id=${run.id}&min_pc=${minPc}&limit=1`).then((p) => p.total),
  elements: (run) => getJSON(`/runs/${run.id}/elements`),
  events: (run, minPc, limit) =>
    getJSON(`/conjunctions?run_id=${run.id}&min_pc=${minPc}&order_by=pc&limit=${limit}`),
};

const cached = {};
const once = (path) => (cached[path] ??= getJSON(path));
const staticSource = {
  async latestRun() {
    const summary = await once("data/summary.json");
    if (!summary.run) throw new HttpError(404, "No screening runs yet");
    return summary.run;
  },
  latestPipeline: async () => (await once("data/summary.json")).pipeline,
  count: async (run, minPc) => (await once("data/summary.json")).counts[minPc] ?? null,
  elements: () => once("data/elements.json"),
  async events(run, minPc, limit) {
    // The export holds the highest-Pc encounters; totals come from the summary.
    const [summary, page] = await Promise.all([once("data/summary.json"), once("data/conjunctions.json")]);
    const items = page.items.filter((c) => c.pc >= Number(minPc));
    return { total: summary.counts[minPc] ?? items.length, items: items.slice(0, limit) };
  },
};

const source = STATIC ? staticSource : apiSource;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;   // never innerHTML: names come from outside
  return node;
}

function fmtPc(pc) {
  if (pc === null || pc === undefined) return "n/a";
  if (pc === 0) return "0";
  const [m, e] = pc.toExponential(1).split("e");
  return `${m} × 10${superscript(e)}`;
}

function superscript(exp) {
  const map = { "-": "⁻", "+": "", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  return [...exp].map((c) => map[c]).join("");
}

function fmtDistance(km) {
  return km < 1 ? `${Math.round(km * 1000)} m` : `${km.toFixed(2)} km`;
}

function fmtRelative(date, now = new Date()) {
  const s = Math.round((date - now) / 1000);
  const abs = Math.abs(s);
  const h = Math.floor(abs / 3600), m = Math.floor((abs % 3600) / 60);
  const span = h > 0 ? `${h} h ${m} min` : m > 0 ? `${m} min` : `${abs} s`;
  return s >= 0 ? `in ${span}` : `${span} ago`;
}

function fmtUtc(date) {
  return date.toISOString().replace("T", " ").slice(0, 19) + " UTC";
}

function showMessage(text) {
  const box = document.getElementById("message");
  box.textContent = text;
  box.hidden = !text;
}

// ------------------------------------------------------------------ globe

const viewer = new Cesium.Viewer("globe", {
  baseLayer: Cesium.ImageryLayer.fromProviderAsync(
    Cesium.TileMapServiceImageryProvider.fromUrl(Cesium.buildModuleUrl("Assets/Textures/NaturalEarthII"))),
  baseLayerPicker: false, geocoder: false, sceneModePicker: false, navigationHelpButton: false,
  fullscreenButton: false, infoBox: false, selectionIndicator: false,
  animation: true, timeline: true, homeButton: true,
});
viewer.scene.globe.enableLighting = true;
viewer.clock.shouldAnimate = true;
viewer.clock.multiplier = 1;

const points = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
const worker = new Worker("worker.js");
let workerReady = false, workerBusy = false;

worker.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "ready") {
    workerReady = true;
  } else if (msg.type === "positions") {
    const p = msg.positions;
    for (let i = 0; i < state.objects.length; i++) {
      const point = state.objects[i].point;
      if (Number.isNaN(p[3 * i])) { point.show = false; continue; }
      point.position = new Cesium.Cartesian3(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
      point.show = true;
    }
    workerBusy = false;
  }
};

setInterval(() => {
  if (!workerReady || workerBusy) return;
  workerBusy = true;
  worker.postMessage({ type: "positions", time: Cesium.JulianDate.toDate(viewer.clock.currentTime).getTime() });
}, UPDATE_MS);

// ------------------------------------------------------------------ data

async function load() {
  let run;
  try {
    run = await source.latestRun();
  } catch (e) {
    showMessage(e.status === 404 ? "No screening runs stored yet. Run `orbitguard pipeline run` first."
                                 : `Could not load the screening data (${e.message}).`);
    return;
  }
  state.run = run;
  showRunWindow();

  const [pipeline, alerts, elements] = await Promise.all([
    source.latestPipeline().catch(() => null),
    source.count(run, ALERT_KEY).catch(() => null),
    source.elements(run).catch((e) => ({ error: e })),
  ]);
  renderStatus(run, pipeline, alerts);

  if (elements.error) {
    showMessage(elements.error.status === 404
      ? "This run was stored without element sets, so its objects cannot be drawn. The next pipeline run will include them."
      : `Could not load element sets (${elements.error.message}).`);
  } else {
    addObjects(elements.objects);
  }
  await loadEvents();
}

function addObjects(objects) {
  const counts = { starlink: 0, oneweb: 0, debris: 0, other: 0 };
  state.objects = objects.map((o) => {
    counts[o.population] += 1;
    const point = points.add({
      position: Cesium.Cartesian3.ZERO, show: false, pixelSize: o.population === "debris" ? 3 : 2,
      color: Cesium.Color.fromCssColorString(COLORS[o.population]),
    });
    const obj = { ...o, point };
    state.index.set(o.norad_id, obj);
    return obj;
  });
  worker.postMessage({ type: "init", tles: objects.map((o) => [o.tle1, o.tle2]) });
  renderLegend(counts);
}

async function loadEvents() {
  const minPc = document.getElementById("threshold").value;
  const list = document.getElementById("events");
  const note = document.getElementById("events-note");
  list.replaceChildren();
  let page;
  try {
    page = await source.events(state.run, minPc, 100);
  } catch (e) {
    note.textContent = `Could not load encounters (${e.message}).`;
    return;
  }
  note.textContent = page.total > page.items.length
    ? `Showing the ${page.items.length} highest of ${page.total.toLocaleString()}.`
    : `${page.total.toLocaleString()} encounter${page.total === 1 ? "" : "s"}.`;
  for (const c of page.items) list.append(eventItem(c));
}

// ------------------------------------------------------------------ panels

function renderStatus(run, pipeline, nAlerts) {
  const box = document.getElementById("status");
  box.replaceChildren();
  const when = new Date(run.generated_at);

  const head = el("div", "status-head");
  head.append(el("span", "label", "Latest screen"), el("span", "value", fmtRelative(when)));
  box.append(head);

  const grid = el("dl", "stats");
  const stat = (k, v) => { grid.append(el("dt", null, k), el("dd", null, v)); };
  stat("Objects screened", run.objects_screened.toLocaleString());
  stat("Close approaches", `${run.n_conjunctions.toLocaleString()} under ${run.threshold_km} km`);
  stat(`Pc ≥ 10⁻⁴`, nAlerts === null ? "n/a" : String(nAlerts));
  stat("Window", `${run.window_hours} h from ${fmtUtc(when)}`);
  // Pc depends on it: until two snapshots exist, the model is an assumption, not a measurement.
  const measured = (run.covariance_source || "").startsWith("calibrated");
  stat("Uncertainty", measured ? "measured from the data" : "assumed: not yet calibrated");
  box.append(grid);

  if (pipeline) {
    const ok = pipeline.status === "success";
    const line = el("p", "pipeline");
    line.append(el("span", `badge ${ok ? "ok" : "bad"}`, ok ? "Pipeline healthy" : `Pipeline ${pipeline.status.replace("_", " ")}`));
    const drift = pipeline.drift_alerts.length;
    line.append(el("span", "muted small", ` last run ${fmtRelative(new Date(pipeline.finished_at))}, `
      + (drift ? `${drift} drift alert${drift > 1 ? "s" : ""}` : "no drift")));
    box.append(line);
  }
}

function renderLegend(counts) {
  const box = document.getElementById("legend");
  box.replaceChildren();
  for (const pop of ["starlink", "oneweb", "debris", "other"]) {
    const row = el("div", "legend-row");
    const dot = el("span", "dot");
    dot.style.background = COLORS[pop];
    row.append(dot, el("span", null, LABELS[pop]), el("span", "count", counts[pop].toLocaleString()));
    box.append(row);
  }
}

function eventItem(c) {
  const li = el("li");
  const button = el("button", "event");
  button.type = "button";
  const pc = el("span", `pc ${c.pc >= ALERT_PC ? "high" : ""}`, fmtPc(c.pc));
  const names = el("span", "names", `${c.name_1} × ${c.name_2}`);
  const meta = el("span", "meta muted small",
    `miss ${fmtDistance(c.miss_distance_km)} · ${c.relative_speed_km_s.toFixed(1)} km/s · TCA ${fmtRelative(new Date(c.tca))}`);
  button.append(pc, names, meta);
  button.addEventListener("click", () => selectEvent(c, button));
  li.append(button);
  return li;
}

// ------------------------------------------------------------------ encounter view

function sampleTrack(satrec, tca) {
  // Coarse samples for the orbit, fine ones around TCA for an exact-looking pass.
  const prop = new Cesium.SampledPositionProperty();
  prop.setInterpolationOptions({ interpolationDegree: 5, interpolationAlgorithm: Cesium.LagrangePolynomialApproximation });
  const offsets = new Set();
  for (let s = -TRACK_MIN * 60; s <= TRACK_MIN * 60; s += 30) offsets.add(s);
  for (let s = -180; s <= 180; s += 1) offsets.add(s);
  for (const s of [...offsets].sort((a, b) => a - b)) {
    const date = new Date(tca.getTime() + s * 1000);
    const pos = ecef(satrec, date);
    if (pos) prop.addSample(Cesium.JulianDate.fromDate(date), pos);
  }
  return prop;
}

function ecef(satrec, date) {
  const pv = satellite.propagate(satrec, date);
  if (!pv || !pv.position) return null;
  const f = satellite.eciToEcf(pv.position, satellite.gstime(date));
  return new Cesium.Cartesian3(f.x * 1000, f.y * 1000, f.z * 1000);
}

function browserClosestApproach(a, b, tca) {
  // Independent check: search +-60 s around the server's TCA, 10 ms steps.
  let best = { km: Infinity, dt: 0 };
  for (let ms = -60000; ms <= 60000; ms += 10) {
    const date = new Date(tca.getTime() + ms);
    const pa = ecef(a, date), pb = ecef(b, date);
    if (!pa || !pb) continue;
    const km = Cesium.Cartesian3.distance(pa, pb) / 1000;
    if (km < best.km) best = { km, dt: ms / 1000 };
  }
  return best;
}

function clearSelection() {
  for (const e of state.selectedEntities) viewer.entities.remove(e);
  state.selectedEntities = [];
  state.encounterPoint = null;
  document.querySelectorAll(".event.active").forEach((b) => b.classList.remove("active"));
  document.getElementById("detail").hidden = true;
  state.selected = null;
}

function selectEvent(c, button) {
  clearSelection();
  const a = state.index.get(c.norad_id_1), b = state.index.get(c.norad_id_2);
  if (!a || !b) {
    showMessage("These objects are not in the stored element sets, so the pass cannot be drawn.");
    return;
  }
  showMessage("");
  state.selected = c;
  button.classList.add("active");

  const tca = new Date(c.tca);
  const satA = satellite.twoline2satrec(a.tle1, a.tle2);
  const satB = satellite.twoline2satrec(b.tle1, b.tle2);
  const entities = [a, b].map((obj, i) => viewer.entities.add({
    name: obj.name,
    position: sampleTrack(i === 0 ? satA : satB, tca),
    point: { pixelSize: 10, color: Cesium.Color.fromCssColorString(COLORS[obj.population]),
             outlineColor: Cesium.Color.WHITE, outlineWidth: 2 },
    label: { text: `${obj.name} (${obj.norad_id})`, font: "13px sans-serif", fillColor: Cesium.Color.WHITE,
             showBackground: true, backgroundColor: Cesium.Color.fromCssColorString("#0b1020cc"),
             pixelOffset: new Cesium.Cartesian2(0, i === 0 ? -22 : 22), scale: 1 },
    path: { leadTime: TRACK_MIN * 60, trailTime: TRACK_MIN * 60, width: 2,
            material: Cesium.Color.fromCssColorString(COLORS[obj.population]).withAlpha(0.7) },
  }));
  state.selectedEntities.push(...entities);

  // Mark the closest approach itself: the pass lasts seconds, the marker stays.
  const pa = ecef(satA, tca), pb = ecef(satB, tca);
  if (pa && pb) {
    state.encounterPoint = Cesium.Cartesian3.midpoint(pa, pb, new Cesium.Cartesian3());
    state.selectedEntities.push(viewer.entities.add({
      position: state.encounterPoint,
      point: { pixelSize: 7, color: Cesium.Color.fromCssColorString("#ff5d5d"),
               outlineColor: Cesium.Color.WHITE, outlineWidth: 1 },
      label: { text: `Closest approach · ${fmtDistance(c.miss_distance_km)}`, font: "12px sans-serif",
               fillColor: Cesium.Color.fromCssColorString("#ffd0d0"), showBackground: true,
               backgroundColor: Cesium.Color.fromCssColorString("#3a0d12cc"),
               pixelOffset: new Cesium.Cartesian2(0, 40) },
    }));
  }

  const start = Cesium.JulianDate.fromDate(new Date(tca.getTime() - TRACK_MIN * 60000));
  const stop = Cesium.JulianDate.fromDate(new Date(tca.getTime() + TRACK_MIN * 60000));
  viewer.timeline.zoomTo(start, stop);
  replay();
  lookAtEncounter();

  const check = browserClosestApproach(satA, satB, tca);
  renderDetail(c, a, b, check);
}

function replay() {
  if (!state.selected) return;
  const tca = new Date(state.selected.tca);
  // Real time from 20 s before: at ~14 km/s the other object crosses the view in seconds.
  viewer.clock.currentTime = Cesium.JulianDate.fromDate(new Date(tca.getTime() - 20000));
  viewer.clock.multiplier = 1;
  viewer.clock.shouldAnimate = true;
}

function lookAtEncounter() {
  // A fixed view of the closest-approach point, so both objects fly through the centre.
  if (!state.encounterPoint) return;
  viewer.camera.flyToBoundingSphere(new Cesium.BoundingSphere(state.encounterPoint, 60000), {
    offset: new Cesium.HeadingPitchRange(0, Cesium.Math.toRadians(-35), 450000), duration: 1.5,
  });
}

function showRunWindow() {
  // The timeline spans the run's screening window; the clock runs in real time.
  const start = Cesium.JulianDate.fromDate(new Date(state.run.generated_at));
  const stop = Cesium.JulianDate.addHours(start, state.run.window_hours, new Cesium.JulianDate());
  viewer.timeline.zoomTo(start, stop);
  viewer.clock.currentTime = Cesium.JulianDate.now();
  viewer.clock.multiplier = 1;
}

function backToGlobe() {
  clearSelection();
  showRunWindow();
  viewer.camera.flyHome(1.5);
}

function renderDetail(c, a, b, check) {
  const box = document.getElementById("detail");
  box.replaceChildren();
  const tca = new Date(c.tca);

  box.append(el("h2", null, "Close approach"));
  const pair = el("div", "pair");
  for (const obj of [a, b]) {
    const row = el("div", "obj");
    const dot = el("span", "dot");
    dot.style.background = COLORS[obj.population];
    row.append(dot, el("span", null, obj.name), el("span", "muted small", ` NORAD ${obj.norad_id} · ${LABELS[obj.population]}`));
    pair.append(row);
  }
  box.append(pair);

  const grid = el("dl", "stats");
  const stat = (k, v, cls) => { grid.append(el("dt", null, k)); grid.append(el("dd", cls || null, v)); };
  stat("Probability of collision", fmtPc(c.pc), c.pc >= ALERT_PC ? "high" : null);
  stat("Miss distance", fmtDistance(c.miss_distance_km));
  stat("Relative speed", `${c.relative_speed_km_s.toFixed(2)} km/s`);
  stat("Closest approach", `${fmtUtc(tca)} (${fmtRelative(tca)})`);
  box.append(grid);

  const countdown = el("p", "countdown");
  box.append(countdown);

  const agree = Math.abs(check.km - c.miss_distance_km) < 0.05;
  box.append(el("p", "muted small",
    `Recomputed in your browser: ${fmtDistance(check.km)} at TCA ${check.dt >= 0 ? "+" : ""}${check.dt.toFixed(2)} s`
    + (agree ? ", matching the server." : ", which differs from the server's value.")));
  box.append(el("p", "muted small",
    "Pc uses position uncertainty measured from successive public element sets for each population; "
    + "with large uncertainty a low Pc can mean \"unknown\" rather than \"safe\"."));

  const actions = el("div", "actions");
  const again = el("button", "btn", "Replay pass");
  again.type = "button";
  again.addEventListener("click", () => { replay(); lookAtEncounter(); });
  const back = el("button", "btn ghost", "Back to globe");
  back.type = "button";
  back.addEventListener("click", backToGlobe);
  actions.append(again, back);
  box.append(actions);
  box.hidden = false;
}

// Simulation time relative to the selected closest approach.
viewer.clock.onTick.addEventListener((clock) => {
  const line = document.querySelector("#detail .countdown");
  if (!state.selected || !line) return;
  const s = (Cesium.JulianDate.toDate(clock.currentTime) - new Date(state.selected.tca)) / 1000;
  line.textContent = Math.abs(s) < 0.5 ? "Closest approach now"
    : `Simulation time: T${s < 0 ? "−" : "+"}${Math.abs(s) < 120 ? Math.abs(s).toFixed(0) + " s"
                                                               : Math.round(Math.abs(s) / 60) + " min"}`;
});

if (STATIC) document.getElementById("api-docs").hidden = true;   // no API behind a static site
document.getElementById("threshold").addEventListener("change", () => { clearSelection(); loadEvents(); });
load();
