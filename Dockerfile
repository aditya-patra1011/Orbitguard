# OrbitGuard API and CLI image.
#   docker build -t orbitguard .
#   docker run -p 8000:8000 -v ./data:/app/data orbitguard

# ---- build stage: install the package and its dependencies into /install
FROM python:3.14-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
# Retry settings ride out slow or flaky connections. They are environment variables,
# not flags, so the separate pip that installs build tools (setuptools) gets them too.
ENV PIP_RETRIES=10 PIP_TIMEOUT=60 PIP_RESUME_RETRIES=10
# The cache mount keeps downloaded wheels between builds, so a code change does not
# re-download ~400 MB.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install ".[ml,api,postgres]"

# ---- runtime stage: no compilers, no source tree, non-root user
FROM python:3.14-slim
# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /install /usr/local
RUN useradd --create-home --uid 1000 orbitguard \
    && mkdir -p /app/data \
    && chown orbitguard:orbitguard /app/data
WORKDIR /app
USER orbitguard

ENV PYTHONUNBUFFERED=1 \
    ORBITGUARD_DB_URL=sqlite:////app/data/orbitguard.db \
    ORBITGUARD_RISK_MODEL=/app/data/models/cdm_risk_model.joblib

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"]
CMD ["orbitguard", "serve", "--host", "0.0.0.0", "--port", "8000"]
