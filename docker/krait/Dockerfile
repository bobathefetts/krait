# Krait (Nemotron 3.5 Super, krait-engine) - runtime image.  Built by stage.sh from a local build; see compose.yml.
# The binaries are the ones krait/build.sh made on the build machine (KRAIT_NATIVE=1 by default: tuned for that CPU).
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-jinja2 python3-regex python3-numpy \
      python3-psutil python3-yaml python3-requests python3-pil python3-jsonschema curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY opt/ /opt/
# runs as an unprivileged user; it may write only its logs and its home
RUN groupadd --system --gid 10001 krait \
    && useradd --system --uid 10001 --gid 10001 --home-dir /opt/krait/home --shell /usr/sbin/nologin krait \
    && mkdir -p /opt/krait/logs /opt/krait/home && chown -R krait:krait /opt/krait/logs /opt/krait/home
ENV KRAIT_ROOT=/opt/krait KRAIT_MODELS=/opt/krait/models KRAIT_PYTHON=python3 HOME=/opt/krait/home HOST=0.0.0.0 \
    PORT=8095 PYTHONDONTWRITEBYTECODE=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER krait
WORKDIR /opt/krait
EXPOSE 8095
# the model goes to /opt/krait/models (read-only mount): models/UD-Q4_K_XL/..., models/mmproj-BF16.gguf
ENTRYPOINT ["/opt/krait/src/krait/krait/krait"]
