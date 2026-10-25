# OrbitGuard

Orbital conjunction screening and ML-based collision risk assessment for objects in Earth orbit.

OrbitGuard pulls the latest public orbital element sets, propagates every object with SGP4,
finds close approaches between all pairs, and scores each one with a probability of collision
(Pc) using position uncertainty **measured** from the data itself. A machine-learning model,
trained on ESA's real conjunction data messages, forecasts an event's final risk two days
before closest approach. Both are served by an HTTP API, packaged with Docker and PostgreSQL.

```
docker compose up -d --build        # PostgreSQL + API on http://localhost:8000 (docs at /docs)
```

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

## Phase 3: forecasting final risk from conjunction data messages

**Data.** ESA's [Collision Avoidance Challenge dataset](https://zenodo.org/records/4463683)
(CC-BY-4.0): 162,634 Conjunction Data Messages (CDMs) that ESA received for its own satellites
in 2015-2019. Each close-approach *event* is a time series of CDMs, refined as closest approach
nears.

**Task** (exactly as in the competition, [Uriot et al. 2020](https://arxiv.org/abs/2008.03069)):
using only CDMs issued at least 2 days before closest approach, predict the risk (log10 Pc) in
the event's final CDM. The score is `L = MSE on high-risk events / F2`, where F2 weights missed
dangerous events (final risk >= 1e-6) four times as heavily as false alarms. Our implementation
reproduces the published baseline exactly (L = 0.694, MSE 0.513, F2 0.739).

```
$ python -m orbitguard risk cv          # cross-validate on training data, choose the threshold
$ python -m orbitguard risk train       # train the final model
$ python -m orbitguard risk evaluate    # score once on the held-out test set
```

**What we learned building it**

1. **Learning from scratch failed badly.** Only 66 of 8,293 training events end high-risk. Every
   model that predicted the final risk directly learned that risky-looking events usually end
   safe, lowered its predictions and lost the recall F2 rewards (cross-validated L 3.4 to 8.7
   versus 0.80 for simply repeating the latest risk).
2. **So the model keeps the latest risk as its backbone** and learns only two corrections: a value
   correction for events it already flags, trained only on events that truly ended high; and a
   classifier for "late risers", low-risk events whose risk still climbs.
3. **Data augmentation** from 3,002 events the strict competition rules discard (predicting each
   event's own last message from earlier ones) raised the late-riser examples from 19 to 69.
4. **The design and threshold were fixed by cross-validation and committed before the test set
   was touched** (pre-registered), with the expectation that the gain was within noise.

**Result on the held-out test set** (2,167 events, 150 high-risk), evaluated once:

| | L | MSE_HR | F2 | Dangerous events missed | False alarms |
|---|---:|---:|---:|---:|---:|
| OrbitGuard model | 0.699 | 0.524 | **0.749** | **31** | 75 |
| Latest-risk baseline | **0.694** | **0.513** | 0.739 | 35 | 63 |
| Constant -5 | 2.504 | 0.679 | 0.271 | 0 | 2,017 |

A statistical tie with a baseline that only 12 of 97 competition teams beat. The late-riser
classifier worked (4 more dangerous events caught for 12 more false alarms); the value correction
slightly hurt, and its top feature, the sunspot number, acts as a time proxy that is fragile
under the shift between the training (0.8% high-risk) and test (6.9%) sets. The winning
team's published rules made their own training score worse while improving the test score, and
30% of teams reported probing the test set; we did not, so these numbers are what a new, unseen
period would look like.

## Phase 4: serving

```
python -m orbitguard screen --group starlink --group fengyun-1c-debris --store   # screen + save
python -m orbitguard db import                                                   # load past results
python -m orbitguard serve                                                       # API on :8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Database and model status, time of the latest run |
| `GET /runs`, `/runs/latest`, `/runs/{id}` | Screening runs and their settings |
| `GET /conjunctions` | Latest run by default; filter by `min_pc`, `max_miss_km`, `norad_id`, TCA window, co-orbital; order by `pc`, `miss` or `tca`; paginated |
| `GET /conjunctions/{id}`, `/objects/{norad_id}/conjunctions` | One event; every event for one object |
| `POST /risk/forecast` | Submit CDM time series, get the Phase 3 forecast of each event's final risk |

**Storage.** SQLAlchemy keeps one code path for SQLite (a single local file, the default) and
PostgreSQL (in Docker), selected by `ORBITGUARD_DB_URL`. Runs and conjunctions are indexed on run,
object, TCA and Pc; a filtered query over a 46,000-conjunction run returns in about 60 ms.

**Docker.** A two-stage, non-root image with a health check; `compose.yaml` runs PostgreSQL 16, the
API and a `cli` service for one-off jobs (`docker compose run --rm cli db import`,
`... cli screen --group oneweb --store`). Data persists in a named volume across restarts.

**Verified, not assumed.**
- An API test sends 40 real ESA test events through `/risk/forecast` and checks the forecasts
  equal the offline evaluation pipeline exactly.
- The same high-risk query gives the same 7 events on SQLite and on PostgreSQL.
- Running the image exposed a real deployment bug: the model would not load because pandas 3 had
  stored strings with pyarrow on the training machine and the image lacked it. The fix (matching
  dependencies, and logging the real error instead of a guess) is in the history.

## Honest limitations

- **The API has no authentication** and the Compose file uses a default database password: it is
  built for local use. Public deployment (Phase 6) needs auth, TLS and secrets management.
- **The ML model does not yet beat the naive forecast.** It ties it, catches more dangerous
  events and loses a little on value accuracy; removing time-proxy features is the next step.
- **CDM data is not available for OrbitGuard's own screening.** CDMs include operator orbit
  determination details that public GP data lacks, so the risk model is a separate component
  for now; the API serves both side by side.
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
| 3 ✅ | ML risk model | ESA CDM data; exact reproduction of the competition metric and baselines; latest-risk backbone with learned value and late-riser corrections; pre-registered, one-time test evaluation |
| 4 ✅ | Serving | Results database (SQLite / PostgreSQL), FastAPI service for conjunctions and risk forecasts, Docker image and Compose stack |
| 5 | Pipeline + MLOps | Scheduled screening and recalibration, data validation, drift monitoring, retraining, CI/CD |
| 6 | Dashboard | 3-D globe (CesiumJS) with live high-risk events, deployed publicly |

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Linux/macOS: .venv/bin/python
                                                  # [dev] includes the [ml] and [api] extras
.venv/Scripts/python -m pytest
```

Data: [CelesTrak](https://celestrak.org) GP data. Please respect its rate limits; the 2-hour cache exists for that reason.
For Phase 3, download `Collision Avoidance Challenge - Dataset.zip` from
[Zenodo record 4463683](https://zenodo.org/records/4463683) and extract it into `data/kelvins/extracted/`.
Tests that need it are skipped when it is absent.
