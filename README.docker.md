# Docker for Zvec

This directory contains Docker configuration for running Zvec in containers.

## Quick Start

### Production Runtime

```bash
# Build the image
docker build -t zvec:latest .

# Run with Docker Compose
docker-compose up -d

# Or run directly
docker run -v $(pwd)/data:/data zvec:latest
```

### Development Environment

```bash
# Start development container with source mounted
docker-compose -f docker-compose.dev.yml up -d

# Enter the container
docker-compose -f docker-compose.dev.yml exec zvec-dev bash

# Inside container, build and install
pip install -e ".[dev]"
pytest python/tests/ -v
```

## Images

| Image | Description |
|-------|-------------|
| `zvec:latest` | Production runtime (manylinux, minimal) |
| `zvec:dev` | Development with all tools |

## Services

| Service | Description | Profile |
|---------|-------------|---------|
| `zvec` | Main database service | default |
| `jupyter` | Jupyter notebook for interactive work | `dev` |
| `zvec-dev` | Live development with source mount | - |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ZVEC_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARN, ERROR) |
| `ZVEC_LOG_TYPE` | `CONSOLE` | Log destination (CONSOLE, FILE) |
| `ZVEC_QUERY_THREADS` | `4` | Query thread pool size |
| `ZVEC_OPTIMIZE_THREADS` | `2` | Background task threads |
| `ZVEC_MEMORY_LIMIT_MB` | `2048` | Memory limit in MB |

## Examples

### Run a Python script with Zvec

```bash
docker run -v $(pwd)/scripts:/app -w /app zvec:latest python my_script.py
```

### Interactive Python shell

```bash
docker run -it -v $(pwd)/data:/data zvec:latest python
```

### Run Jupyter with Zvec

```bash
docker-compose --profile dev up -d
# Open http://localhost:8888
```

## Volumes

- `zvec-data`: Persistent storage for vector collections
- `zvec-build`: Build cache (dev only)
- `pip-cache`: Python package cache (dev only)
