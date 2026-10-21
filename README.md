# OrbitGuard

Orbital conjunction screening and ML-based collision risk assessment for objects in Earth orbit.

OrbitGuard pulls the latest public orbital element sets, propagates every object with SGP4,
finds close approaches between all pairs, and scores each one with a probability of collision
(Pc) using position uncertainty **measured** from the data itself.

## What works today

```
$ python -m orbitguard calibrate          # measure prediction error from cached snapshots
$ python -m orbitguard screen --group starlink --group oneweb \
      --group cosmos-2251-debris --group iridium-33-debris --group fengyun-1c-debris --hours 24

45948 conjunctions < 5 km (+64 co-orbital pairs flying in formation)
Pc: covariance 'calibrated 2026-10-21 from 13730 element-set pairs', hard-body radius 10 m; 7 events with Pc >= 1e-4

TCA (UTC)             miss km v_rel km/s        Pc  objects
2026-10-22 08:17:08    0.196      14.69  1.64e-03  FENGYUN 1C DEB (30266) x FENGYUN 1C DEB (37928)
2026-10-22 19:16:29    0.059      14.41  1.00e-03  FENGYUN 1C DEB (30336) x FENGYUN 1C DEB (30493)
2026-10-22 07:07:50    1.184       0.90  1.75e-04  FENGYUN 1C DEB (29876) x FENGYUN 1C DEB (32117)
...
```

## Pipeline

```
CelesTrak GP data ──► validate + cache ──► SGP4 catalog ──► vectorised propagation (10 s grid)
                       (2 h TTL, 403-safe)   (drop stale       │
                                              element sets)    ▼
                                            KD-tree pair search per step, radius = d + v_max·Δt/2
                                                               │  ~1.4 M pairs / 2 h
                                                               ▼
                                            linear relative-motion pre-filter (vectorised)
                                                               │  ~1 % survive
                                                               ▼
                                            group into encounters ──► bounded TCA refinement
                                                               │
       cached snapshots ──► calibrate ──► per-population       ▼
       (same objects,       RTN errors    covariance ──► encounter-plane Pc (2-D, HBR disc)
        days apart)         vs. age        models              │
                                                               ▼
                                            conjunctions: TCA, miss, v_rel, Pc ──► JSON
```

## Phase 1: screening design decisions

- **No missed encounters by construction.** Two objects can close at up to ~15.8 km/s, so the
  KD-tree radius is padded by `v_max · Δt / 2`. A test checks a head-on crossing is still caught
  with a 60 s grid.
- **Linear pre-filter is lossless in practice.** On a live 1 h window, filtered and unfiltered
  runs produced the identical 1,773 conjunctions; the filter made the pipeline ~6x faster.
- **Stale data is dropped.** Element sets older than 14 days are excluded rather than producing
  meaningless miss distances.
- **Formation flyers are separated.** Pairs with relative speed < 0.1 km/s (same-plane Starlinks,
  docked vehicles) are flagged `co_orbital` and get no Pc: the short-encounter model does not
  apply to them.

## Phase 2: collision probability

**Pc method.** Standard short-encounter 2-D Pc (Foster 1992, Alfano 2005, as used by NASA CARA):
project the relative position and the combined covariance of both objects onto the plane
perpendicular to the relative velocity at TCA, then integrate that 2-D Gaussian over a disc of
the combined hard-body radius (HBR). Tests check it against the exact closed form (to 1e-6) and
a 2-million-sample Monte Carlo.

**Covariance calibration.** Public GP data has no covariance, so OrbitGuard measures it. For
every object present in two cached snapshots, the older element set is propagated to the newer
one's epoch and compared in the radial / along-track / cross-track (RTN) frame. A robust (MAD)
spread per age bin is fitted with `sigma = sigma0 + growth * age`.

First calibration: 13,730 element-set pairs, 2026-10-19 vs 2026-10-21:

| Population | Pairs | Outliers (manoeuvres) | Along-track 1σ after 2 days |
|---|---:|---:|---:|
| Debris | 1,964 | 4.9 % | **0.75 km** |
| OneWeb | 651 | 7.5 % | 1.24 km |
| Starlink | 11,112 | 9.3 % | **38.07 km** |
| *Assumed default* | | | *2.50 km* |

Two findings drove the design:

1. **One model does not fit all.** Starlink satellites manoeuvre almost continuously with electric
   propulsion, so their public element sets go stale ~50x faster than passive debris. A pooled
   model (81 % Starlink) would be ~30x too pessimistic for debris, so OrbitGuard fits **one model
   per population**.
2. **Calibration changes the answer.** With measured uncertainty, events above the usual action
   threshold (Pc ≥ 1e-4) fall from 97 to 7, and the list becomes almost entirely
   debris-on-debris, where orbits are known well enough for a close pass to stay a close pass.

## Honest limitations

- **Low Pc can mean "unknown", not "safe".** With 38 km of along-track uncertainty, a Starlink
  pair's Pc is diluted to near zero whatever the true geometry (the *dilution region*). Operators
  solve this with their own precise ephemerides; public data cannot. Phase 3 addresses risk
  estimation under exactly this kind of uncertainty.
- **The calibration is young.** Two snapshots ~2 days apart give ages clustered around 2 days, so
  the error at epoch (`sigma0`) is not yet separable from growth and fits to ~0 (floored at 1 m).
  Each new screening run caches a snapshot, so the fit improves as data accumulates.
- **The reference is itself an estimate.** The newer element set has its own error, so measured
  spreads are slightly conservative.
- **Hard-body radius is assumed** (10 m combined) because object sizes are not used yet.
- CelesTrak's `active` group excludes most debris; complete screening needs the full catalog
  (Space-Track, free account).

## Roadmap

| Phase | Goal | Key pieces |
|---|---|---|
| 1 ✅ | Screening engine | Ingestion + cache, SGP4, KD-tree screening, TCA refinement, tests |
| 2 ✅ | Collision probability | 2-D encounter-plane Pc; RTN covariance model calibrated per population from successive element sets |
| 3 | ML risk model | Train on ESA Kelvins Collision Avoidance Challenge CDM data: predict the *final* risk of an event from early CDMs; baseline vs. gradient boosting vs. sequence model; MLflow tracking |
| 4 | Serving | FastAPI (`/conjunctions`, `/risk/{event}`), Postgres/TimescaleDB, Docker |
| 5 | Pipeline + MLOps | Scheduled screening and recalibration, data validation, drift monitoring, retraining, CI/CD |
| 6 | Dashboard | 3-D globe (CesiumJS) with live high-risk events, deployed publicly |

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest
```

Data: [CelesTrak](https://celestrak.org) GP data. Please respect its rate limits; the 2-hour cache exists for that reason.
