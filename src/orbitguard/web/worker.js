// Propagates the whole catalogue off the main thread so the globe stays smooth.
// Same SGP4 model as the server; positions are returned Earth-fixed, in metres.
importScripts("https://cdn.jsdelivr.net/npm/satellite.js@5.0.0/dist/satellite.min.js");

let sats = [];

onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "init") {
    sats = msg.tles.map(([line1, line2]) => satellite.twoline2satrec(line1, line2));
    postMessage({ type: "ready", count: sats.length });
  } else if (msg.type === "positions") {
    const date = new Date(msg.time);
    const gmst = satellite.gstime(date);
    const out = new Float64Array(sats.length * 3);
    for (let i = 0; i < sats.length; i++) {
      const pv = satellite.propagate(sats[i], date);
      if (!pv || !pv.position) {   // decayed or failed: hide it
        out[3 * i] = NaN;
        continue;
      }
      const f = satellite.eciToEcf(pv.position, gmst);
      out[3 * i] = f.x * 1000;
      out[3 * i + 1] = f.y * 1000;
      out[3 * i + 2] = f.z * 1000;
    }
    postMessage({ type: "positions", time: msg.time, positions: out }, [out.buffer]);
  }
};
