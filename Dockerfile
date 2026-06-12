# Reproducible image for the biomllab toolchain.
# Uses the official uv image so the container build matches local + CI tooling.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Faster, quieter, reproducible installs.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching) using only the lockfiles.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv pip install --system -e ".[dev]"

# Copy the rest of the project (tests, configs).
COPY . .

# Default: prove the toolchain works inside the container.
# Use the system interpreter directly (deps already installed above) so we don't
# spin up a second venv at runtime.
CMD ["python", "-m", "pytest"]
