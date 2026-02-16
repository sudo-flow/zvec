# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zvec is an in-process vector database built on Proxima (Alibaba's vector search engine). It consists of:
- **C++ Core** (`src/`): Vector search algorithms (HNSW, IVF, Flat), indexing, storage, and SQL query engine
- **Python Bindings** (`python/`): PyBind11 wrapper exposing the C++ core to Python
- **Alilego Library** (`src/ailego/`): Internal utility library for math, containers, I/O, and parallelism

## Build System

The project uses CMake with scikit-build-core for Python wheels. Key CMake options:
- `BUILD_PYTHON_BINDINGS=ON`: Enable Python bindings (default: OFF for CMake-only builds)
- `CMAKE_BUILD_TYPE`: Debug/Release/Coverage
- Architecture flags: `-DENABLE_SKYLAKE=ON`, `-DENABLE_ZEN3=ON`, etc. for CPU-specific optimizations

## Development Commands

### Python Development
```bash
# Editable install (builds C++ extension in-place)
pip install -e ".[dev]"

# Run tests
pytest python/tests/ -v

# Run tests with coverage
pytest python/tests/ --cov=zvec --cov-report=term-missing

# Format and lint
ruff check python/
ruff format python/
```

### C++ Development (standalone)
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

## Architecture

### C++ Core Structure
- `src/core/algorithm/`: Vector index implementations
  - `hnsw/`: Hierarchical Navigable Small World (dense vectors)
  - `hnsw_sparse/`: HNSW for sparse vectors
  - `flat/`: Brute-force flat search
  - `flat_sparse/`: Flat search for sparse vectors
  - `ivf/`: Inverted File index
  - `cluster/`: K-means clustering
- `src/db/`: Database layer
  - `index/`: Column indexing (vector and inverted), storage (WAL, Arrow IPC, Parquet), segments
  - `sqlengine/`: SQL query parsing and analysis
  - `common/`: Configuration, error handling, logging, cgroup utilities
- `src/binding/python/`: PyBind11 bindings (compiled as `_zvec` module)

### Python Layer Structure
- `python/zvec/zvec.py`: Entry point functions (`init`, `create_and_open`, `open`)
- `python/zvec/model/`: Core Python classes
  - `collection.py`: `Collection` class (wraps `_Collection` from C++)
  - `doc.py`: `Doc` class for document representation
  - `schema/`: Schema definitions (`CollectionSchema`, `FieldSchema`, `VectorSchema`)
  - `param/`: Parameter classes for index/query configuration
- `python/zvec/executor/`: Query execution framework
- `python/zvec/extension/`: Embedding and reranking functions (OpenAI, Qwen, Sentence Transformers, BM25)

### Key Pattern: Python-C++ Integration
The Python wrapper layer (`python/zvec/model/`) wraps C++ objects exposed via PyBind11:
- C++ objects are accessed via private `_obj` attribute
- Schema updates (e.g., after `create_index`) refresh the cached `_schema`
- Conversion between Python and C++ representations happens in `convert.py`

## Testing

- Python tests: `python/tests/` (organized by module: `test_collection.py`, `test_doc.py`, etc.)
- Detailed tests: `python/tests/detail/` (low-level collection operations)
- C++ tests: `tests/` (using Google Test framework)

## Code Quality

- **Python**: Ruff for linting/formatting (see `pyproject.toml` `[tool.ruff]`)
- **C++**: C++17 standard, `-Wall -Werror=return-type` enforced
- Type stubs are generated but stored separately (see `stub/` in sdist)

## Platform Support

- Linux (x86_64, ARM64) - primary development platform
- macOS (ARM64) - supported
- Uses platform-specific SIMD optimizations via architecture-specific CMake flags

## Important Notes

- The Python bindings are built as the `_zvec` module (note underscore prefix)
- `zvec.init()` must be called before any other operations (sets up logging, thread pools, memory limits)
- Collection operations are thread-safe but individual `Collection` objects should not be shared across threads
- Vector indexes can only be created on vector fields; inverted indexes are for scalar fields
