# CLAUDE.md — Xe

## What This Is

Go library for Intel Xe GPU compute via Level Zero. Two execution models: direct in-process and daemon-based (works around Intel's IGC JIT compiler bugs). Zero-copy shared memory between Go and GPU.

## Build

```bash
CGO_ENABLED=1 go build ./...   # Linux with Level Zero
CGO_ENABLED=0 go build ./...   # stubs on non-Linux
go test -v ./...
```

Daemon (production path):
```bash
cd xe-daemon && make
```

## Architecture

- `xe.go` — direct mode: Level Zero init, shared memory alloc, SPIR-V dispatch, matmul/RMSNorm/softmax
- `xe_stub.go` — no-op stubs for non-Linux
- `daemon.go` — daemon mode: Unix socket protocol, zero-copy arena (256MB memfd), L3 bridge, CE buffers
- `daemon_stub.go` — stubs for non-Linux

## Key Design

- **Daemon mode** is production — IGC JIT segfaults ~50% of the time inside Go's address space. Daemon process has different memory layout that avoids the bug.
- **Zero-copy arena** — 256MB memfd split: Go region (128MB) + guard page + Xe region (128MB). Both sides mmap the same fd.
- **L3 bridge** — cache-coherent host memory for CUDA↔Xe data sharing through CPU L3 (~10ns latency vs PCIe).
- **SPIR-V kernels** — compiled from GLSL/OpenCL C, dispatched via Level Zero command lists.

## Related Packages

- `github.com/open-ai-org/mongoose` — GPU compute engine (Xe is one backend)
