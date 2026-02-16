# =============================================================================
# Multi-stage Dockerfile for Zvec
# Produces a minimal runtime image with Python bindings
# =============================================================================

# Stage 1: Build stage - manylinux for maximum compatibility
FROM quay.io/pypa/manylinux_2_28_x86_64 AS builder

# Install build dependencies including CMake
RUN yum install -y \
    git \
    openssl-devel \
    && yum clean all

# Install CMake binary directly (not via pip)
RUN curl -sSL https://github.com/Kitware/CMake/releases/download/v3.30.0/cmake-3.30.0-linux-x86_64.tar.gz -o cmake.tar.gz \
    && tar -xzf cmake.tar.gz --strip-components=1 -C /usr/local \
    && rm cmake.tar.gz

# Set Python version
ENV PATH="/opt/python/cp310-cp310/bin:${PATH}"

# Install Python build tools (ninja via pip is fine)
RUN pip install --no-cache-dir \
    scikit-build-core>=0.11.0 \
    "pybind11>=2.13.0" \
    ninja>=1.11 \
    setuptools_scm>=8.0

# Verify CMake is available
RUN cmake --version

# Set working directory
WORKDIR /build

# Copy source code (including .git for setuptools_scm and submodule init)
COPY . .

# Initialize git submodules - this downloads the third-party dependencies
RUN git submodule update --init --recursive || \
    echo "Warning: Some submodules may have failed to initialize"

# Build the wheel
RUN pip wheel . --no-deps -w /dist --no-build-isolation \
    --config-settings='cmake.define.BUILD_PYTHON_BINDINGS="ON"' \
    --config-settings='cmake.define.BUILD_TOOLS="OFF"' \
    --config-settings='cmake.define.CMAKE_BUILD_TYPE="Release"' \
    --config-settings='install.strip=true'

# Stage 2: Runtime stage - minimal image
FROM quay.io/pypa/manylinux_2_28_x86_64

# Install runtime dependencies
RUN yum install -y \
    # Vector operations (OpenMP)
    libgomp \
    # General utilities
    ca-certificates \
    && yum clean all \
    && rm -rf /var/cache/yum

# Create non-root user
RUN groupadd -r zvec && useradd -r -g zvec -u 1000 -m -s /bin/bash zvec

# Set Python version
ENV PATH="/opt/python/cp310-cp310/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install numpy (only runtime dependency)
RUN pip install --no-cache-dir numpy>=1.23

# Copy and install the built wheel
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Create data directory
RUN mkdir -p /data && chown -R zvec:zvec /data

# Switch to non-root user
USER zvec
WORKDIR /data

# Expose health check port (for future HTTP API)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import zvec; zvec.init(); exit(0)" || exit 1

# Default command - start a Python shell
CMD ["python"]
