# Docker Technical Documentation

**Image**: `mayotte-xas:1.0.0`  
**Base**: `python:3.10-slim` (Debian Trixie)  
**Author**: Charles Le Losq  
**Last updated**: November 2025

## Overview

This document provides technical details about the Docker implementation for the Mayotte XAS analysis pipeline. For general usage, see [README.md](USAGE.md).

---

## Table of Contents

- [Docker Architecture](#docker-architecture)
- [Dockerfile Breakdown](#dockerfile-breakdown)
- [docker-run.sh Script](#docker-runsh-script)
- [Volume Mounts](#volume-mounts)
- [Build Process](#build-process)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Docker Architecture

### Why Docker?

**Reproducibility**:
- Identical environment across Windows, macOS, Linux
- Fixed Python version (3.10)
- Locked package versions
- No dependency conflicts

**Fortran Integration**:
- gfortran automatically installed
- ctsfg6 compiled during image build
- Executable available to Python scripts
- No manual compilation needed by users

**Isolation**:
- Clean environment (no interference with system Python)
- No risk of breaking existing projects
- Easy cleanup (just delete image)

### Image Specifications

```
Base Image:   python:3.10-slim
Size:         ~800 MB (compressed)
OS:           Debian Trixie (testing)
Architecture: x86_64, arm64 (Apple Silicon)
```

**Installed Components**:
- Python 3.10.x with pip
- gfortran (Fortran compiler)
- System libraries: HDF5, OpenBLAS, LAPACK
- Python packages (see requirements.txt)
- Compiled ctsfg6 executable

---

## Dockerfile Breakdown

### Stage 1: Base Image & System Dependencies

```dockerfile
FROM python:3.10-slim

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \              # C compiler (for Python extensions)
    g++ \              # C++ compiler
    gfortran \         # Fortran compiler (for ctsfg6)
    libhdf5-dev \      # HDF5 (required by h5py, used by Larch)
    libopenblas-dev \  # BLAS library (linear algebra)
    liblapack-dev \    # LAPACK (linear algebra)
    pkg-config \       # Build configuration
    && rm -rf /var/lib/apt/lists/*
```

**Note**: `libopenblas-dev` replaces `libatlas-base-dev` (not available in Debian Trixie)

### Stage 2: User Setup (Security)

```dockerfile
# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash xasuser && \
    mkdir -p /home/xasuser/mayotte && \
    chown -R xasuser:xasuser /home/xasuser

WORKDIR /home/xasuser/mayotte
```

**Security Best Practice**: Never run containers as root

### Stage 3: Python Dependencies

```dockerfile
# Copy requirements first (Docker layer caching)
COPY --chown=xasuser:xasuser requirements.txt .

# Install Python packages
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt
```

**Caching Strategy**: requirements.txt copied separately to leverage Docker layer caching (rebuilds faster if only code changes)

### Stage 4: Code & Fortran Compilation

```dockerfile
# Copy all source code
COPY --chown=xasuser:xasuser . .

# Compile Fortran code
RUN cd src && \
    gfortran ctsfg6.for -o ctsfg6 && \
    chmod +x ctsfg6 && \
    cd ..
```

**Compilation Details**:
- ctsfg6.for: 2632 lines of Fortran 77 code
- No optimization flags (default -O0 for debuggability)
- Output: executable `src/ctsfg6`

### Stage 5: Output Directories

```dockerfile
# Create output directories
RUN mkdir -p figures/Iron figures/Sulfur figures/Modelling results/modelling && \
    chown -R xasuser:xasuser figures results
```

### Stage 6: User & Entrypoint

```dockerfile
USER xasuser
WORKDIR /home/xasuser/mayotte/src

ENTRYPOINT []
CMD ["python", "analysis_publication.py"]
```

**Key Points**:
- WORKDIR = `src/` (scripts run from here)
- Empty ENTRYPOINT allows flexible commands
- Default CMD runs XAS analysis

---

## docker-run.sh Script

### Purpose

Wrapper script providing user-friendly interface to Docker commands:
- `build` - Build image
- `run` - Run XAS analysis
- `model` - Run thermodynamic modelling
- `all` - Run complete pipeline
- `exec` - Interactive shell
- `stop`, `clean`, `logs` - Management commands

### Key Functions

#### build_image()
```bash
docker build -t mayotte-xas:1.0.0 .
```

#### run_analysis()
```bash
docker run --rm \
    --name mayotte_xas_analysis \
    -v "$(pwd)/xas:/home/xasuser/mayotte/xas:ro" \
    -v "$(pwd)/tables/liste.xlsx:/home/xasuser/mayotte/tables/liste.xlsx:ro" \
    -v "$(pwd)/figures:/home/xasuser/mayotte/figures" \
    -v "$(pwd)/results:/home/xasuser/mayotte/results" \
    mayotte-xas:1.0.0
```

**Volume Mounts**:
- `xas/`: Read-only (input data)
- `tables/liste.xlsx`: Read-only (metadata)
- `figures/`: Read-write (output)
- `results/`: Read-write (output)
- `src/`: NOT mounted (code is in image)

#### run_modelling()
```bash
docker run --rm \
    --name mayotte_xas_analysis_model \
    -v "$(pwd)/figures:/home/xasuser/mayotte/figures" \
    -v "$(pwd)/results:/home/xasuser/mayotte/results" \
    mayotte-xas:1.0.0 python modelling.py
```

**Minimal mounts** (only needs results/ as input)

#### run_all()
```bash
docker run --rm \
    --name mayotte_xas_analysis_all \
    -v "$(pwd)/xas:/home/xasuser/mayotte/xas:ro" \
    -v "$(pwd)/tables/liste.xlsx:/home/xasuser/mayotte/tables/liste.xlsx:ro" \
    -v "$(pwd)/figures:/home/xasuser/mayotte/figures" \
    -v "$(pwd)/results:/home/xasuser/mayotte/results" \
    mayotte-xas:1.0.0 \
    bash -c "python analysis_publication.py && python modelling.py"
```

**Sequential execution** with visual separator

---

## Volume Mounts

### Read-Only Mounts (`:ro`)

**Purpose**: Protect input data from accidental modification

```bash
-v "$(pwd)/xas:/home/xasuser/mayotte/xas:ro"
-v "$(pwd)/tables/liste.xlsx:/home/xasuser/mayotte/tables/liste.xlsx:ro"
```

**Container Path**: `/home/xasuser/mayotte/xas`  
**Host Path**: `./xas` (current directory on host)

### Read-Write Mounts

**Purpose**: Allow scripts to write output

```bash
-v "$(pwd)/figures:/home/xasuser/mayotte/figures"
-v "$(pwd)/results:/home/xasuser/mayotte/results"
```

**Behavior**: Files created in container appear on host immediately

### Why src/ is NOT Mounted

**Reason**: Code is baked into the image

**Advantages**:
- Immutable code (reproducibility)
- No path issues
- Simpler volume management
- ctsfg6 executable always in correct location

**Development**: Use `exec` to test changes without rebuilding

---

## Build Process

### Command

```bash
./docker-run.sh build
```

### What Happens

1. **Context**: Docker reads `Dockerfile` and `.dockerignore`
2. **Base Image**: Pull `python:3.10-slim` from Docker Hub (~150 MB)
3. **System Packages**: Install gcc, gfortran, libraries (~300 MB)
4. **Python Packages**: Install numpy, scipy, larch, etc. (~200 MB)
5. **Code Copy**: Copy project files (~5 MB)
6. **Fortran Compilation**: Compile ctsfg6 (~5 seconds)
7. **Image Creation**: Final image tagged as `mayotte-xas:1.0.0`

**Total Time**: 3-10 minutes (depends on internet speed)

### Build Output

```
[+] Building 187.3s (15/15) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load .dockerignore
 => [1/10] FROM python:3.10-slim
 => [2/10] RUN apt-get update && apt-get install -y...
 => [3/10] RUN useradd -m -u 1000...
 => [4/10] COPY --chown=xasuser:xasuser requirements.txt .
 => [5/10] RUN pip install...
 => [6/10] COPY --chown=xasuser:xasuser . .
 => [7/10] RUN cd src && gfortran ctsfg6.for -o ctsfg6...
 => [8/10] RUN mkdir -p figures/Iron...
 => exporting to image
 => => naming to docker.io/library/mayotte-xas:1.0.0
```

### Verification

```bash
# Check image exists
docker images | grep mayotte-xas

# Should show:
# mayotte-xas   1.0.0   abc123def456   2 minutes ago   856MB
```

---

## Troubleshooting

### Build Failures

#### "E: Package 'libatlas-base-dev' has no installation candidate"

**Cause**: libatlas-base-dev removed from Debian Trixie

**Solution**: Already fixed in current Dockerfile (uses libopenblas-dev)

```dockerfile
libopenblas-dev \  # ✅ Use this
# libatlas-base-dev \  # ❌ Don't use this
```

#### "ERROR [7/10] RUN cd src && gfortran ctsfg6.for -o ctsfg6"

**Cause**: Fortran compilation error

**Debug**:
```bash
# Build with debug output
docker build --progress=plain -t mayotte-xas:1.0.0 .

# Check ctsfg6.for syntax
docker run --rm -v $(pwd)/src:/src python:3.10-slim bash -c "
  apt-get update && apt-get install -y gfortran && 
  cd /src && gfortran -c ctsfg6.for
"
```

#### "No space left on device"

**Cause**: Insufficient Docker disk space

**Solution**:
```bash
# Clean up old images
docker system prune -a

# Or increase Docker Desktop disk allocation (Settings > Resources)
```

### Runtime Failures

#### "FileNotFoundError: ./xas/iron/sample.dat not found"

**Cause**: Volume mount incorrect or file missing

**Debug**:
```bash
# Check volume mounts
docker run --rm -v $(pwd)/xas:/data alpine ls -lh /data/iron/

# Should list .dat files
```

#### "FileNotFoundError: ctsfg6 not found"

**Cause**: ctsfg6 not compiled or wrong working directory

**Debug**:
```bash
# Check ctsfg6 exists in image
docker run --rm mayotte-xas:1.0.0 ls -lh /home/xasuser/mayotte/src/ctsfg6

# Should show executable with +x permission
```

#### "Permission denied" on output files

**Cause**: File ownership mismatch (host UID ≠ container UID 1000)

**Solution**:
```bash
# Fix ownership on host
sudo chown -R $USER:$USER figures/ results/

# Or run container with host UID (advanced)
docker run --rm --user $(id -u):$(id -g) ...
```

### Docker Daemon Issues

#### "Cannot connect to Docker daemon"

**Cause**: Docker Desktop not running

**Solution**:
- Start Docker Desktop application
- Wait for status indicator to turn green

#### "docker: command not found"

**Cause**: Docker not installed

**Solution**:
- Install Docker Desktop: https://www.docker.com/products/docker-desktop

---

## Advanced Usage

### Interactive Shell

```bash
# Open bash shell in container
./docker-run.sh exec

# Inside container:
cd /home/xasuser/mayotte/src
python  # Start Python REPL
ls -lh ctsfg6  # Check executable
exit
```

### Custom Python Script

```bash
docker run --rm \
    -v $(pwd)/figures:/home/xasuser/mayotte/figures \
    -v $(pwd)/results:/home/xasuser/mayotte/results \
    mayotte-xas:1.0.0 python -c "
import numpy as np
print('NumPy version:', np.__version__)
"
```

### Debug Mode

```bash
# Run with verbose output
docker run --rm \
    -v $(pwd)/xas:/home/xasuser/mayotte/xas:ro \
    -v $(pwd)/tables/liste.xlsx:/home/xasuser/mayotte/tables/liste.xlsx:ro \
    -v $(pwd)/figures:/home/xasuser/mayotte/figures \
    -v $(pwd)/results:/home/xasuser/mayotte/results \
    mayotte-xas:1.0.0 python -u analysis_publication.py
```

**`-u` flag**: Unbuffered Python output (see prints immediately)

### Mount Source Code (Development)

```bash
# Override src/ with host version (for testing changes)
docker run --rm \
    -v $(pwd)/src:/home/xasuser/mayotte/src \
    -v $(pwd)/xas:/home/xasuser/mayotte/xas:ro \
    -v $(pwd)/tables/liste.xlsx:/home/xasuser/mayotte/tables/liste.xlsx:ro \
    -v $(pwd)/figures:/home/xasuser/mayotte/figures \
    -v $(pwd)/results:/home/xasuser/mayotte/results \
    mayotte-xas:1.0.0 python analysis_publication.py
```

**Warning**: ctsfg6 must be compiled on host

### Resource Limits

```bash
# Limit CPU and memory
docker run --rm \
    --cpus="2.0" \
    --memory="4g" \
    -v $(pwd)/xas:/home/xasuser/mayotte/xas:ro \
    -v $(pwd)/tables/liste.xlsx:/home/xasuser/mayotte/tables/liste.xlsx:ro \
    -v $(pwd)/figures:/home/xasuser/mayotte/figures \
    -v $(pwd)/results:/home/xasuser/mayotte/results \
    mayotte-xas:1.0.0
```

### Multi-Container (Docker Compose)

**docker-compose.yml** (example):
```yaml
version: '3.8'

services:
  xas-analysis:
    image: mayotte-xas:1.0.0
    container_name: mayotte_xas
    volumes:
      - ./xas:/home/xasuser/mayotte/xas:ro
      - ./tables/liste.xlsx:/home/xasuser/mayotte/tables/liste.xlsx:ro
      - ./figures:/home/xasuser/mayotte/figures
      - ./results:/home/xasuser/mayotte/results
    command: python analysis_publication.py
    
  modelling:
    image: mayotte-xas:1.0.0
    container_name: mayotte_model
    depends_on:
      - xas-analysis
    volumes:
      - ./figures:/home/xasuser/mayotte/figures
      - ./results:/home/xasuser/mayotte/results
    command: python modelling.py
```

**Usage**:
```bash
docker-compose up
```

---

## Image Maintenance

### Update Python Packages

```bash
# Edit requirements.txt (add new package or update version)

# Rebuild image (no cache for pip install)
docker build --no-cache-filter pip -t mayotte-xas:1.0.0 .
```

### Update System Packages

```bash
# Edit Dockerfile (add new apt package)

# Full rebuild
docker build --no-cache -t mayotte-xas:1.0.0 .
```

### Tag New Version

```bash
# Build with new tag
docker build -t mayotte-xas:1.1.0 .

# Keep old version
docker tag mayotte-xas:1.0.0 mayotte-xas:1.0.0-backup

# Update docker-run.sh IMAGE_TAG variable
```

### Export/Import Image

```bash
# Save image to file
docker save mayotte-xas:1.0.0 | gzip > mayotte-xas-1.0.0.tar.gz

# Transfer file to another machine, then:
docker load < mayotte-xas-1.0.0.tar.gz
```

---

## Performance Optimization

### Build Caching

**Current Strategy**:
- requirements.txt copied separately (layer caching)
- Code copied last (changes don't rebuild dependencies)

**Result**: Subsequent builds after code changes: ~30 seconds vs. 5 minutes

### Slim Base Image

**python:3.10-slim** vs. **python:3.10**:
- Slim: ~150 MB base
- Full: ~900 MB base
- Savings: ~750 MB

**Trade-off**: Need to manually install system packages

### Multi-Stage Build (Future Enhancement)

```dockerfile
# Stage 1: Build dependencies
FROM python:3.10-slim as builder
RUN pip install --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim
COPY --from=builder /root/.local /root/.local
# Copy code and compile Fortran
...
```

**Benefit**: Smaller final image (no build tools in runtime)

---

## Security Considerations

### Non-Root User

✅ Container runs as `xasuser` (UID 1000), not root

**Why**: Limits damage if container is compromised

### Read-Only Mounts

✅ Input data mounted with `:ro` flag

**Why**: Prevents accidental data corruption

### Minimal Attack Surface

✅ Slim base image (no unnecessary packages)
✅ No SSH, no shells beyond bash
✅ No network exposure (EXPOSE commented out)

### Supply Chain

⚠️ Base image from Docker Hub (trusted, but not audited)
⚠️ Python packages from PyPI (official, but anyone can affect PyPI, beware of mispelling package names)

---

## Comparison: Docker vs. Local

| Aspect | Docker | Local |
|--------|--------|-------|
| **Setup Time** | 5-10 min (once) | 30-60 min |
| **Reproducibility** | Perfect | Variable |
| **Windows Support** | Yes | No (Fortran issue) |
| **Isolation** | Complete | None |
| **Performance** | ~5% overhead | Native |
| **Debugging** | Moderate | Easy |
| **Updates** | Rebuild image | pip install |

---

## References

- **Docker Official Docs**: https://docs.docker.com/
- **Python Docker Images**: https://hub.docker.com/_/python
- **Dockerfile Best Practices**: https://docs.docker.com/develop/dev-best-practices/
- **Docker Volumes**: https://docs.docker.com/storage/volumes/

---

**Last updated**: November 2025  
**Maintainer**: Charles Le Losq  
**Docker Version**: 24.0.6+
