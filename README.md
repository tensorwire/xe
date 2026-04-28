# Xe

Go library for Intel Xe GPU compute via [Level Zero](https://spec.oneapi.io/level-zero/latest/index.html). Provides two execution models — direct in-process access and a daemon-based architecture that works around Intel's IGC JIT compiler bugs.

Built for ML training and inference workloads. Dispatches SPIR-V compute kernels with zero-copy shared memory between Go and the GPU.

## Requirements

- **Linux** (Intel Level Zero is Linux-only)
- **Intel Xe GPU** — Arc discrete or integrated (Arrow Lake, Meteor Lake, etc.)
- **Level Zero runtime** — `libze_loader.so`
- **Go 1.25+** with CGO enabled

### Installing Level Zero

```bash
# Ubuntu/Debian
sudo apt install intel-level-zero-gpu level-zero-dev

# Fedora
sudo dnf install level-zero level-zero-devel

# From source
git clone https://github.com/oneapi-src/level-zero.git
cd level-zero && mkdir build && cd build
cmake .. && make -j$(nproc) && sudo make install
```

## Architecture

Xe provides two ways to use Intel GPUs from Go:

### Direct Mode (`Xe`)

```go
gpu := xe.NewXe()
if gpu == nil {
    log.Fatal("no Intel GPU found")
}
defer gpu.Sync()

result := gpu.MatMul(a, b, m, k, n)
gpu.RMSNorm(data, weights, 1e-5)
gpu.SoftMax(logits, vocabSize)
```

Direct mode initializes Level Zero inside the Go process. Simple API for basic compute — matmul, RMSNorm, softmax, ReLU. Uses shared memory (unified memory on integrated GPUs) so data is accessible from both CPU and GPU without explicit copies.

**Caveat:** Intel's IGC (Intel Graphics Compiler) JIT has a known bug where it crashes ~50% of the time when running inside Go's address space. The memory layout of the Go runtime triggers a segfault in the compiler. Direct mode probes with an external C binary first (`xe-probe`) and only proceeds if hardware is confirmed working.

### Daemon Mode (`Daemon`)

```go
d := xe.NewDaemon()
if d == nil {
    log.Fatal("no Intel GPU or xe-daemon not found")
}
defer d.Close()

// Load a SPIR-V compute kernel
idx := d.LoadSPIRV("/path/to/kernel.spv", "rmsnorm")

// Allocate GPU-accessible shared memory
ptr := d.Alloc(1024 * 4) // 1024 floats
defer d.Free(ptr)

// Dispatch and sync
d.DispatchRMSNorm(idx, x, out, weight, dim, seqLen, 1e-5)
d.Sync()
```

Daemon mode is the production path. It starts `xe-daemon` — a standalone C process that holds the Level Zero context. Go communicates via Unix socket for commands and shared memory for data. The C process has a different memory layout that doesn't trigger the IGC bug.

**Build the daemon:**

```bash
cd xe-daemon
make
sudo make install  # optional — installs to /usr/local/bin
```

The Go library searches for `xe-daemon` in these locations (in order):
1. `./xe-daemon/xe-daemon`
2. `./xe-daemon`
3. Same directory as the calling binary
4. `/usr/local/bin/xe-daemon`

## Zero-Copy Shared Memory

### Split Arena

The daemon allocates a 256 MB shared memory arena via `memfd_create` — anonymous, no filesystem footprint. Both Go and the daemon mmap the same file descriptor.

```
 0 MB ┌──────────────────────┐
      │                      │
      │    GO REGION         │  Go writes here (logits, targets, input data)
      │    128 MB            │
      │                      │
128MB ├──────────────────────┤
      │  GUARD (4KB)         │  mprotect(PROT_NONE) — segfaults on access
128MB ├──────────────────────┤
 +4KB │                      │
      │    XE REGION         │  Xe GPU writes here (losses, gradients, outputs)
      │    ~128 MB           │
      │                      │
256MB └──────────────────────┘
```

```go
d := xe.NewDaemon()
if d.HasArena() {
    // Write input data — zero copy, Go writes directly to shared pages
    logits := d.GoRegion(0, batchSize*vocabSize)
    copy(logits, myLogits)

    targets := d.GoRegionInt32(batchSize*vocabSize*4, batchSize)
    copy(targets, myTargets)

    // After GPU dispatch + sync, read results from Xe region
    d.Sync()
    losses := d.XeRegion(xe.ArenaXeStart, batchSize)
}
```

### Cross-Entropy Buffers

For cross-entropy loss computation, the daemon provides dedicated L0 shared memory buffers. These are `zeMemAllocShared` allocations — the GPU's page table maps them directly, so both CPU and GPU access the same physical pages on integrated GPUs.

```go
ce := d.AllocCEBuffers(seqLen, vocabSize)

// Write logits and targets — directly into GPU-accessible memory
logits := ce.CELogitsFloat32(seqLen * vocabSize)
copy(logits, modelOutput)

targets := ce.CETargetsInt32(seqLen)
copy(targets, labels)

// Dispatch cross-entropy on Xe
d.DispatchCrossEntropy(kernelIdx, 0, 0, 0, 0, uint32(seqLen), uint32(vocabSize), 1.0/float32(seqLen))
d.Sync()

// Read losses and gradients — zero copy
losses := ce.CELossesFloat32(seqLen)
grads := ce.CEGradFloat32(seqLen * vocabSize)
```

### L3 Cache Bridge (CUDA + Xe)

For systems with both NVIDIA and Intel GPUs, the L3 bridge provides cache-coherent host memory that both GPUs can access through the CPU's L3 cache.

```go
bridge := d.GetL3Bridge()
if bridge != nil {
    // Xe allocates the memory via zeMemAllocHost (cached)
    // CUDA registers it via cudaHostRegister
    // Both GPUs see the same data through CPU L3 coherency

    activations := bridge.L3Float32(0, tensorSize)
    // CUDA writes activations via DMA → lands in L3
    // Xe reads from L3 → no PCIe round-trip
}
```

The bridge is 64 MB by default — enough for ~50 cached activation tensors during training. Data flows through the CPU's cache coherency protocol, not PCIe, so latency is determined by L3 access time (~10ns) rather than PCIe bandwidth.

## SPIR-V Kernel Dispatch

### Direct Mode

```go
spirvBytes, _ := os.ReadFile("rmsnorm.spv")
idx := xe.LoadSPIRV(spirvBytes, "rmsnorm")

xe.DispatchRMSNorm(idx, xPtr, outPtr, weightPtr, dim, seqLen, 1e-5)
xe.DispatchSiLUGate(idx2, gatePtr, upPtr, outPtr, n)
xe.DispatchAddInPlace(idx3, aPtr, bPtr, n)
xe.SyncAll()
```

### Daemon Mode

```go
idx := d.LoadSPIRV("/path/to/rmsnorm.spv", "rmsnorm")
d.DispatchRMSNorm(idx, xPtr, outPtr, weightPtr, dim, seqLen, 1e-5)
d.Sync()
```

Supported kernel dispatches:
- **RMSNorm** — `out = rmsnorm(x, weight, eps)`, one workgroup per sequence position
- **SiLU-gate-mul** — `out[i] = gate[i] * sigmoid(gate[i]) * up[i]`
- **Add in-place** — `a[i] += b[i]`
- **Cross-entropy** — forward + backward in one dispatch
- **FP32→BF16 conversion** — on L3 bridge data

All dispatches use workgroup size 256. Kernels are SPIR-V binaries compiled from GLSL or OpenCL C via `glslangValidator` or `clang`.

## Non-Linux Builds

On non-Linux platforms (macOS, Windows), all functions return nil/zero/no-op. The package compiles everywhere — it just does nothing without an Intel GPU and Level Zero.

```go
gpu := xe.NewXe()       // returns nil on macOS
d := xe.NewDaemon()     // returns nil on macOS
xe.Initialized()        // returns false on macOS
```

## Daemon Protocol

The daemon speaks a line-based text protocol over a Unix socket:

| Command | Response | Description |
|---------|----------|-------------|
| `info` | `OK <name> <mem> <kernels>` | Device info |
| `load <path> <entry>` | `OK <idx>` | Load SPIR-V kernel |
| `alloc <bytes>` | `OK <ptr>` | Allocate shared memory |
| `free <ptr>` | `OK` | Free memory |
| `rmsnorm <idx> <x> <out> <w> <dim> <seq> <eps>` | `OK` | Dispatch RMSNorm |
| `silu <idx> <gate> <up> <out> <n>` | `OK` | Dispatch SiLU |
| `add <idx> <a> <b> <n>` | `OK` | Dispatch add |
| `cebufs <npos> <vocab>` | `OK <logits> <targets> <losses> <grad>` | Allocate CE buffers |
| `crossentropy <idx> <npos> <vocab> <inv_n>` | `OK` | Dispatch cross-entropy |
| `convert_bf16 <idx> <fp32_off> <bf16_off> <n>` | `OK` | FP32→BF16 on L3 bridge |
| `l3bridge` | `OK <ptr> <size>` | Get L3 bridge pointer |
| `arena` | `OK <size> <half> <xe_start>` | Get arena layout |
| `sync` | `OK` | Wait for GPU completion |
| `quit` | `OK` | Shutdown daemon |

## API Reference

### Direct Mode

| Function | Description |
|----------|-------------|
| `NewXe() *Xe` | Initialize Intel GPU. Returns nil if unavailable. |
| `(x *Xe) Name() string` | Device name (e.g., `"xe/Intel Arc A770"`) |
| `(x *Xe) VRAM() uint64` | Total shared memory in bytes |
| `(x *Xe) MatMul(a, b []float32, m, k, n int) []float32` | Matrix multiply on shared memory |
| `(x *Xe) RMSNorm(data, weight []float32, eps float32)` | In-place RMS normalization |
| `(x *Xe) SoftMax(data []float32, n int)` | In-place softmax |
| `(x *Xe) ReLU(data []float32)` | In-place ReLU |
| `(x *Xe) Sync()` | Wait for GPU ops |
| `SharedAlloc(bytes int) unsafe.Pointer` | Allocate L0 shared memory |
| `Free(ptr unsafe.Pointer)` | Free L0 memory |
| `Initialized() bool` | Check if Level Zero is ready |
| `LoadSPIRV(data []byte, name string) int` | Load SPIR-V kernel, returns index |
| `DispatchRMSNorm(idx int, x, out, w unsafe.Pointer, dim, seq int, eps float32)` | Dispatch RMSNorm kernel |
| `DispatchSiLUGate(idx int, gate, up, out unsafe.Pointer, n int)` | Dispatch SiLU-gate-mul |
| `DispatchAddInPlace(idx int, a, b unsafe.Pointer, n int)` | Dispatch a += b |
| `SyncAll()` | Global GPU sync |

### Daemon Mode

| Function | Description |
|----------|-------------|
| `NewDaemon() *Daemon` | Start xe-daemon, connect via Unix socket. Returns nil if unavailable. |
| `(d *Daemon) Close()` | Shutdown daemon, unmap arena, cleanup |
| `(d *Daemon) Name() string` | Device name |
| `(d *Daemon) VRAM() uint64` | Shared memory size |
| `(d *Daemon) LoadSPIRV(path, entry string) int` | Load SPIR-V from file |
| `(d *Daemon) Alloc(bytes int) unsafe.Pointer` | Allocate shared memory |
| `(d *Daemon) Free(ptr unsafe.Pointer)` | Free shared memory |
| `(d *Daemon) DispatchRMSNorm(...)` | Dispatch RMSNorm |
| `(d *Daemon) DispatchSiLU(...)` | Dispatch SiLU-gate-mul |
| `(d *Daemon) DispatchAdd(...)` | Dispatch add |
| `(d *Daemon) Sync()` | Wait for GPU ops |
| `(d *Daemon) HasArena() bool` | Check if zero-copy arena is available |
| `(d *Daemon) GoRegion(off, n int) []float32` | Float32 slice in Go write region |
| `(d *Daemon) GoRegionInt32(off, n int) []int32` | Int32 slice in Go write region |
| `(d *Daemon) XeRegion(off, n int) []float32` | Float32 slice in Xe write region (read after Sync) |
| `(d *Daemon) GetL3Bridge() *L3Bridge` | Get CUDA↔Xe cache bridge |
| `(d *Daemon) AllocCEBuffers(nPos, vocab int) *CEBuffers` | Allocate cross-entropy buffers |
| `(d *Daemon) DispatchCrossEntropy(...)` | Dispatch cross-entropy kernel |
| `(d *Daemon) ConvertFP32ToBF16(idx int, fp32Off, bf16Off, n int) error` | BF16 conversion on L3 bridge |

## License

MIT
