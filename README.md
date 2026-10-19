# OrbitGuard

Orbital conjunction screening and ML-based collision risk assessment for objects in Earth orbit.

OrbitGuard pulls the latest public orbital element sets, propagates every object with SGP4,
finds close approaches between all pairs, and (in progress) scores how risky each one really is.

## What works today (Phase 1: screening engine)

```
$ python -m orbitguard screen --group starlink --group oneweb \
      --group cosmos-2251-debris --group iridium-33-debris --group fengyun-1c-debris --hours 24

Screened 14391 objects over 24 h in 273.0 s
47753 conjunctions < 5 km (+61 co-orbital pairs flying in formation)

TCA (UTC)             miss km v_rel km/s  objects
2026-10-20 18:21:46    0.031      12.08  COSMOS 2251 DEB (34446) x FENGYUN 1C DEB (35105)
2026-10-20 05:32:56    0.038       7.20  STARLINK-34500 (64613) x STARLINK-36966 (68044)
...
```

## How screening works

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
                                                               ▼
                                            conjunctions: TCA, miss distance, relative speed,
                                            co-orbital flag  ──► JSON
```

Design decisions worth knowing:

- **No missed encounters by construction.** Two objects can close at up to ~15.8 km/s, so the
  KD-tree radius is padded by `v_max · Δt / 2`. A test checks a head-on crossing is still caught
  with a 60 s grid.
- **Linear pre-filter is lossless in practice.** On a live 1 h window, filtered and unfiltered
  runs produced the identical 1,773 conjunctions; the filter made the pipeline ~6x faster.
- **Stale data is dropped.** SGP4 error grows ~1-3 km/day, so element sets older than 14 days
  are excluded rather than producing meaningless miss distances.
- **Formation flyers are separated.** Pairs with relative speed < 0.1 km/s (same-plane Starlinks,
  docked vehicles) are flagged `co_orbital` instead of flooding the conjunction list.

## Honest limitations

- **TLE/GP data has no covariance and km-level error.** A 31 m predicted miss is *not* a 31 m
  real miss. This is exactly why Phase 2 exists.
- CelesTrak's `active` group excludes most debris; real screening needs the full catalog
  (Space-Track, free account) or the debris groups shown above.
- Operators manoeuvre; public data lags those manoeuvres.

## Roadmap

| Phase | Goal | Key pieces |
|---|---|---|
| 1 ✅ | Screening engine | Ingestion + cache, SGP4, KD-tree screening, TCA refinement, tests |
| 2 | Collision probability | Pc via Foster/Chan 2-D method in the B-plane, with covariance estimated from TLE-history residuals |
| 3 | ML risk model | Train on ESA Kelvins Collision Avoidance Challenge CDM data: predict the *final* risk of an event from early CDMs; baseline vs. gradient boosting vs. sequence model; MLflow tracking |
| 4 | Serving | FastAPI (`/conjunctions`, `/risk/{event}`), Postgres/TimescaleDB, Docker |
| 5 | Pipeline + MLOps | Scheduled screening every few hours, data validation, drift monitoring, retraining, CI/CD |
| 6 | Dashboard | 3-D globe (CesiumJS) with live high-risk events, deployed publicly |

## Development

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest
```

Data: [CelesTrak](https://celestrak.org) GP data. Please respect its rate limits; the 2-hour cache exists for that reason.
