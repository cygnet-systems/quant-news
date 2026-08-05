FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first (smaller image).
#
# 2.10.0, not 2.2.0. The old pin silently disabled two of the six models in
# production while the run still reported success:
#   - transformers 5.x requires torch >= 2.4 and disables the PyTorch backend
#     otherwise, so DeBERTa could not load (1 of 20 symbols scored).
#   - torch 2.2 was built against NumPy 1.x, and the image resolves NumPy 2.x,
#     so torch's array bridge failed ("_ARRAY_API not found") and Kronos threw
#     on every symbol (0 of 20).
# Both models still reported ready=True, because is_ready() does not exercise
# them. Keep this in step with the version the project runs locally.
RUN pip install --no-cache-dir \
    torch==2.10.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HF_HOME must be set BEFORE the downloads so they land where the runtime
# looks. Previously it was set after, and the download passed
# cache_dir=/app/models_cache — which makes that path the cache ROOT, while
# HF_HOME expects the hub cache one level down at $HF_HOME/hub. The weights
# were baked into the image and then never found, so Kronos failed on every
# symbol in production with "cannot find the requested files in the local
# cache". One variable now governs both sides.
ENV HF_HOME=/app/models_cache

# Pre-bake every model weight — zero runtime downloads. DeBERTa was missing
# here entirely, so it could not load offline either; it returned HOLD
# without scoring anything and looked like a working model.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('NeoQuasar/Kronos-mini'); \
snapshot_download('NeoQuasar/Kronos-Tokenizer-base'); \
snapshot_download('mrm8488/deberta-v3-ft-financial-news-sentiment-analysis')"

# Only AFTER the downloads — set earlier, they would block the build's own
# fetches.
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

COPY . .
RUN mkdir -p /app/cache/trained_models /app/cache/dash_bg_callbacks

COPY scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8050
ENTRYPOINT ["/entrypoint.sh"]
# Single worker is mandatory: models, caches, and the scheduler are
# process-local. --loop asyncio because plotly_cloud's nest_asyncio hook
# cannot patch uvloop. --host 0.0.0.0 so the published port is reachable
# (config.py defaults HOST to 127.0.0.1).
CMD ["uvicorn", "app:server", "--host", "0.0.0.0", "--port", "8050", "--workers", "1", "--loop", "asyncio"]
