# Strata on Nemotron 3.5 Super (strata-nh-engine) - runtime image.  Built by stage.sh from a local build; see compose.yml.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-jinja2 python3-regex python3-numpy \
      python3-psutil python3-yaml python3-requests python3-pil python3-jsonschema curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY opt/ /opt/
ENV NH_ROOT=/opt/strata NH_PYTHON=python3 HOST=0.0.0.0 PORT=8095 PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /opt/strata
EXPOSE 8095
# the model goes to /opt/strata/models (read-only mount), as the config names it: models/UD-Q4_K_XL/..., models/mmproj-BF16.gguf
ENTRYPOINT ["/opt/strata/src/strata/nemotron/run-nemotron.sh"]
