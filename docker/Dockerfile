# syntax=docker/dockerfile:1
# MARL-BattleGrounds container image: Python 3.14.5, the locked packages, the
# browser clients and the trained Season 0 models, ready to run.
#
# Build from the repository root (.dockerignore keeps everything else out):
#   docker build -f docker/Dockerfile -t marl-battlegrounds .                       # CPU
#   docker build -f docker/Dockerfile --build-arg BUILD=cuda12 -t marl-battlegrounds:cuda12 .
# Published images are built by the release workflow from the release tag:
#   ghcr.io/arm-2-lab/marl-battlegrounds:<version>         (CPU; amd64, arm64)
#   ghcr.io/arm-2-lab/marl-battlegrounds:<version>-cuda12  (NVIDIA; amd64)
# How to run them: docs/install.md. docker/entrypoint.sh starts the commands.
#
# Build arguments:
#   PYTHON_VERSION  Python patch, default 3.14.5: the version the Season 0
#                   models were trained and tested with. A training run saved
#                   in this image can be extended in this image.
#   BUILD           cpu (default), cuda12 or cuda13. rocm and tpu images are
#                   not supported: they need software from the host machine.
#   SEASON_0_MODELS 1 (default) downloads the six trained Season 0 models and
#                   the Season 0 training-opponent pool into the image from
#                   Hugging Face, so a read-only image (Apptainer) can train
#                   against them offline; 0 skips them (they then download on
#                   first use).
# An optional build secret hf_token (--secret id=hf_token,src=FILE) lets the
# download read private Hugging Face repositories; it never enters the image.
ARG PYTHON_VERSION=3.14.5
FROM ghcr.io/astral-sh/uv:0.11.16 AS uv

FROM python:${PYTHON_VERSION}-slim-trixie
ARG BUILD=cpu
ARG SEASON_0_MODELS=1

# tini runs as process 1 and stops the relays with the game; socat is the
# relay that lets Docker's published ports reach the 127.0.0.1-only clients.
RUN apt-get update \
    && apt-get install -y --no-install-recommends socat tini \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv

# The environment and the model cache live under /opt, readable by every user
# (the entrypoint runs as the owner of the mounted /work folder).
ENV UV_PROJECT_ENVIRONMENT=/opt/marl-bgs/venv \
    UV_PYTHON=/usr/local/bin/python3 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    HF_HUB_CACHE=/opt/marl-bgs/models \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    MARL_BGS_PORT=8765 \
    MARL_BGS_REPLAY_PORT=8767 \
    VIRTUAL_ENV=/opt/marl-bgs/venv \
    PATH=/opt/marl-bgs/venv/bin:$PATH

WORKDIR /opt/marl-bgs/source
COPY pyproject.toml uv.lock README.md LICENSE .python-version ./
RUN extra=""; if [ "$BUILD" != cpu ]; then extra="--extra $BUILD"; fi \
    && uv sync --locked --no-install-project --extra all --group jupyter $extra
COPY src ./src
COPY web ./web
RUN extra=""; if [ "$BUILD" != cpu ]; then extra="--extra $BUILD"; fi \
    && uv sync --locked --no-editable --extra all --group jupyter $extra \
    && python -c "import marl_battlegrounds"
RUN --mount=type=secret,id=hf_token,env=HF_TOKEN,required=false \
    mkdir -p "$HF_HUB_CACHE" \
    && if [ "$SEASON_0_MODELS" = 1 ]; then \
        python -m marl_battlegrounds models download \
            MARL-BattleGrounds/tdm-season-0-recurrent-mappo \
            MARL-BattleGrounds/tdm-season-0-recurrent-ippo \
            MARL-BattleGrounds/tdm-season-0-feedforward-mappo \
            MARL-BattleGrounds/tdm-season-0-feedforward-ippo \
            MARL-BattleGrounds/tdm-season-0-qmix \
            MARL-BattleGrounds/tdm-season-0-pqn-vdn \
            MARL-BattleGrounds/tdm-season-0-results/training-opponents; \
    fi \
    && chmod -R a+rwX "$HF_HUB_CACHE" \
    && rm -rf /root/.cache
COPY --chmod=0755 play.py docker/entrypoint.sh /opt/marl-bgs/

WORKDIR /work
EXPOSE 8766 8768
# Docker's stop sends Ctrl-C, so a game being played saves its replay first.
STOPSIGNAL SIGINT
ENTRYPOINT ["tini", "--", "/opt/marl-bgs/entrypoint.sh"]
CMD ["play"]
