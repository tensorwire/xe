//go:build linux

package xe

/*
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <linux/memfd.h>

// Create anonymous shared memory via memfd_create.
// Returns fd, or -1 on error. No filesystem footprint.
static int xe_memfd_create(int size) {
    int fd = syscall(SYS_memfd_create, "xe-arena", MFD_CLOEXEC);
    if (fd < 0) return -1;
    if (ftruncate(fd, size) < 0) { close(fd); return -1; }
    return fd;
}

// Clear the close-on-exec flag so the fd survives exec into the daemon.
static void xe_clear_cloexec(int fd) {
    int flags = fcntl(fd, F_GETFD);
    fcntl(fd, F_SETFD, flags & ~1);
}
*/
import "C"

import (
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// Arena layout constants for the split shared memory region.
//
//	[0 .. 128MB)           GO REGION   — Go writes logits + targets
//	[128MB .. 128MB+4KB)   GUARD       — mprotect(PROT_NONE), segfaults on access
//	[128MB+4KB .. 256MB)   XE REGION   — Xe writes losses + gradients
const (
	ArenaSize    = 256 * 1024 * 1024
	ArenaHalf    = ArenaSize / 2
	ArenaGuard   = 4096
	ArenaXeStart = ArenaHalf + ArenaGuard
)

// Daemon manages the xe-daemon child process and communicates via Unix socket.
// The daemon holds the Level Zero context in a separate C process to avoid
// the IGC JIT compiler crash that occurs in Go's address space.
//
// Data exchange uses a split shared memory arena (memfd-backed) for zero-copy
// bulk operations, plus L0 shared memory for cross-entropy buffers.
type Daemon struct {
	conn    net.Conn
	proc    *exec.Cmd
	sock    string
	name    string
	memSize uint64
	mu      sync.Mutex

	arenaFd   int
	arenaData []byte
}

// NewDaemon starts the xe-daemon and connects via Unix socket.
// Returns nil if xe-daemon binary not found or no Intel GPU.
func NewDaemon() *Daemon {
	paths := []string{
		"./xe-daemon/xe-daemon",
		"./xe-daemon",
		filepath.Join(filepath.Dir(os.Args[0]), "xe-daemon", "xe-daemon"),
		"/usr/local/bin/xe-daemon",
	}
	var binPath string
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			binPath = p
			break
		}
	}
	if binPath == "" {
		return nil
	}

	stale, _ := filepath.Glob("/tmp/xe-*.sock")
	for _, s := range stale {
		os.Remove(s)
	}

	arenaFd := int(C.xe_memfd_create(C.int(ArenaSize)))
	if arenaFd >= 0 {
		C.xe_clear_cloexec(C.int(arenaFd))
	}

	sock := fmt.Sprintf("/tmp/xe-%d.sock", os.Getpid())
	var cmd *exec.Cmd
	if arenaFd >= 0 {
		cmd = exec.Command(binPath, sock, fmt.Sprintf("%d", arenaFd))
		cmd.ExtraFiles = []*os.File{os.NewFile(uintptr(arenaFd), "arena")}
	} else {
		cmd = exec.Command(binPath, sock)
	}
	cmd.Stderr = os.Stderr

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil
	}

	if err := cmd.Start(); err != nil {
		return nil
	}

	scanner := bufio.NewScanner(stdout)
	var name string
	var memSize uint64
	deadline := time.After(5 * time.Second)
	readyCh := make(chan bool, 1)
	go func() {
		if scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "READY ") {
				fields := strings.Fields(line[6:])
				if len(fields) >= 2 {
					name = strings.Join(fields[:len(fields)-1], " ")
					memSize, _ = strconv.ParseUint(fields[len(fields)-1], 10, 64)
				}
				readyCh <- true
				return
			}
		}
		readyCh <- false
	}()

	select {
	case ok := <-readyCh:
		if !ok {
			cmd.Process.Kill()
			return nil
		}
	case <-deadline:
		cmd.Process.Kill()
		return nil
	}

	conn, err := net.Dial("unix", sock)
	if err != nil {
		cmd.Process.Kill()
		return nil
	}

	d := &Daemon{
		conn:    conn,
		proc:    cmd,
		sock:    sock,
		name:    name,
		memSize: memSize,
	}

	if arenaFd >= 0 {
		data, err := syscall.Mmap(arenaFd, 0, ArenaSize,
			syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
		if err != nil {
			log.Printf("[xe] arena mmap failed: %v", err)
		} else {
			d.arenaFd = arenaFd
			d.arenaData = data
			log.Printf("[xe] arena: %d MB zero-copy (go: 0-%dMB, xe: %dMB-%dMB)",
				ArenaSize/(1024*1024), ArenaHalf/(1024*1024),
				ArenaXeStart/(1024*1024), ArenaSize/(1024*1024))
		}
	}

	log.Printf("[xe] daemon connected: %s (%d MB shared)", name, memSize/1024/1024)

	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		<-sigCh
		d.Close()
		os.Exit(0)
	}()

	return d
}

func (d *Daemon) cmd(format string, args ...interface{}) string {
	d.mu.Lock()
	defer d.mu.Unlock()
	msg := fmt.Sprintf(format, args...) + "\n"
	d.conn.Write([]byte(msg))
	buf := make([]byte, 4096)
	n, _ := d.conn.Read(buf)
	return strings.TrimSpace(string(buf[:n]))
}

// Close shuts down the daemon, unmaps the arena, and cleans up.
func (d *Daemon) Close() {
	if d.conn == nil {
		return
	}
	if d.arenaData != nil {
		syscall.Munmap(d.arenaData)
		d.arenaData = nil
	}
	if d.arenaFd > 0 {
		syscall.Close(d.arenaFd)
		d.arenaFd = 0
	}
	d.cmd("quit")
	d.conn.Close()
	d.conn = nil
	if d.proc != nil && d.proc.Process != nil {
		d.proc.Process.Kill()
		d.proc.Wait()
	}
	os.Remove(d.sock)
}

func (d *Daemon) Name() string  { return "xe/" + d.name }
func (d *Daemon) VRAM() uint64  { return d.memSize }

// LoadSPIRV loads a SPIR-V kernel from file. Returns kernel index.
func (d *Daemon) LoadSPIRV(path, entryPoint string) int {
	resp := d.cmd("load %s %s", path, entryPoint)
	if strings.HasPrefix(resp, "OK ") {
		idx, _ := strconv.Atoi(resp[3:])
		return idx
	}
	return -1
}

// Alloc allocates shared memory on the Xe device.
func (d *Daemon) Alloc(bytes int) unsafe.Pointer {
	resp := d.cmd("alloc %d", bytes)
	if strings.HasPrefix(resp, "OK ") {
		addr, _ := strconv.ParseUint(strings.TrimPrefix(resp[3:], "0x"), 16, 64)
		return unsafe.Pointer(uintptr(addr))
	}
	return nil
}

// Free releases Xe shared memory.
func (d *Daemon) Free(ptr unsafe.Pointer) {
	d.cmd("free %p", ptr)
}

// DispatchRMSNorm dispatches RMSNorm on the Xe GPU.
func (d *Daemon) DispatchRMSNorm(kernelIdx int, x, out, weight unsafe.Pointer, dim, seqLen int, eps float32) {
	d.cmd("rmsnorm %d %p %p %p %d %d %f", kernelIdx, x, out, weight, dim, seqLen, eps)
}

// DispatchSiLU dispatches fused SiLU-gate-mul on Xe.
func (d *Daemon) DispatchSiLU(kernelIdx int, gate, up, out unsafe.Pointer, n int) {
	d.cmd("silu %d %p %p %p %d", kernelIdx, gate, up, out, n)
}

// DispatchAdd dispatches element-wise add on Xe.
func (d *Daemon) DispatchAdd(kernelIdx int, a, b unsafe.Pointer, n int) {
	d.cmd("add %d %p %p %d", kernelIdx, a, b, n)
}

// Sync waits for all Xe GPU ops to complete.
func (d *Daemon) Sync() {
	d.cmd("sync")
}

// L3Bridge is cache-coherent host memory shared between CUDA and Xe through CPU L3 cache.
// CUDA writes via DMA (lands in L3). Xe reads from L3 directly. Zero latency interconnect.
type L3Bridge struct {
	Ptr  unsafe.Pointer
	Size int
}

// GetL3Bridge returns the L3 cache bridge allocated by the daemon.
// The pointer is zeMemAllocHost memory — Xe can access it directly.
// CUDA accesses it after cudaHostRegister on the Go/CUDA side.
func (d *Daemon) GetL3Bridge() *L3Bridge {
	resp := d.cmd("l3bridge")
	if !strings.HasPrefix(resp, "OK ") {
		return nil
	}
	var ptr uintptr
	var size int
	fmt.Sscanf(resp[3:], "%p %d", &ptr, &size)
	if ptr == 0 {
		return nil
	}
	log.Printf("[xe] L3 bridge: %d MB cache-coherent", size/(1024*1024))
	return &L3Bridge{Ptr: unsafe.Pointer(ptr), Size: size}
}

// L3Float32 returns a float32 slice view into the L3 bridge at a byte offset.
func (b *L3Bridge) L3Float32(byteOffset, count int) []float32 {
	return (*[1 << 28]float32)(unsafe.Pointer(uintptr(b.Ptr) + uintptr(byteOffset)))[:count:count]
}

// L3Uint16 returns a uint16 slice view into the L3 bridge at a byte offset.
func (b *L3Bridge) L3Uint16(byteOffset, count int) []uint16 {
	return (*[1 << 28]uint16)(unsafe.Pointer(uintptr(b.Ptr) + uintptr(byteOffset)))[:count:count]
}

// ConvertFP32ToBF16 dispatches the fp32_to_bf16 SPIR-V kernel on Xe.
func (d *Daemon) ConvertFP32ToBF16(kernelIdx int, fp32Offset, bf16Offset, n int) error {
	resp := d.cmd("convert_bf16 %d %d %d %d", kernelIdx, fp32Offset, bf16Offset, n)
	if resp == "OK" {
		return nil
	}
	return fmt.Errorf("convert_bf16: %s", resp)
}

// HasArena returns true if zero-copy shared memory is available.
func (d *Daemon) HasArena() bool {
	return d.arenaData != nil
}

// GoRegion returns a float32 slice in the Go write region (below ArenaHalf).
func (d *Daemon) GoRegion(byteOffset, count int) []float32 {
	end := byteOffset + count*4
	if end > ArenaHalf {
		panic(fmt.Sprintf("xe arena: Go write at %d+%d exceeds Go region (%d)", byteOffset, count*4, ArenaHalf))
	}
	return (*[1 << 28]float32)(unsafe.Pointer(&d.arenaData[byteOffset]))[:count:count]
}

// GoRegionInt32 returns an int32 slice in the Go write region.
func (d *Daemon) GoRegionInt32(byteOffset, count int) []int32 {
	end := byteOffset + count*4
	if end > ArenaHalf {
		panic(fmt.Sprintf("xe arena: Go write at %d+%d exceeds Go region (%d)", byteOffset, count*4, ArenaHalf))
	}
	return (*[1 << 28]int32)(unsafe.Pointer(&d.arenaData[byteOffset]))[:count:count]
}

// XeRegion returns a float32 slice in the Xe write region (above ArenaXeStart).
// Go reads from here after Sync().
func (d *Daemon) XeRegion(byteOffset, count int) []float32 {
	if byteOffset < ArenaXeStart {
		panic(fmt.Sprintf("xe arena: Xe read at %d below Xe region (%d)", byteOffset, ArenaXeStart))
	}
	return (*[1 << 28]float32)(unsafe.Pointer(&d.arenaData[byteOffset]))[:count:count]
}

// CEBuffers holds L0 shared memory pointers for cross-entropy.
// Go writes directly to these. Xe reads directly. Zero copies.
type CEBuffers struct {
	Logits  unsafe.Pointer
	Targets unsafe.Pointer
	Losses  unsafe.Pointer
	Grad    unsafe.Pointer
}

// AllocCEBuffers allocates L0 shared memory for cross-entropy on Xe.
func (d *Daemon) AllocCEBuffers(nPos, vocabSize int) *CEBuffers {
	resp := d.cmd("cebufs %d %d", nPos, vocabSize)
	if !strings.HasPrefix(resp, "OK ") {
		log.Printf("[xe] cebufs failed: %s", resp)
		return nil
	}
	var lp, tp, op, gp uintptr
	fmt.Sscanf(resp[3:], "%p %p %p %p", &lp, &tp, &op, &gp)
	if lp == 0 {
		return nil
	}
	return &CEBuffers{
		Logits:  unsafe.Pointer(lp),
		Targets: unsafe.Pointer(tp),
		Losses:  unsafe.Pointer(op),
		Grad:    unsafe.Pointer(gp),
	}
}

// CELogitsFloat32 returns the logits buffer as a Go float32 slice.
func (b *CEBuffers) CELogitsFloat32(n int) []float32 {
	return (*[1 << 28]float32)(b.Logits)[:n:n]
}

// CETargetsInt32 returns the targets buffer as a Go int32 slice.
func (b *CEBuffers) CETargetsInt32(n int) []int32 {
	return (*[1 << 28]int32)(b.Targets)[:n:n]
}

// CELossesFloat32 returns the losses buffer as a Go float32 slice.
func (b *CEBuffers) CELossesFloat32(n int) []float32 {
	return (*[1 << 28]float32)(b.Losses)[:n:n]
}

// CEGradFloat32 returns the gradient buffer as a Go float32 slice.
func (b *CEBuffers) CEGradFloat32(n int) []float32 {
	return (*[1 << 28]float32)(b.Grad)[:n:n]
}

// DispatchCrossEntropy fires the Xe cross-entropy kernel.
// Write to CEBuffers BEFORE calling. After dispatch, call Sync() then read results.
func (d *Daemon) DispatchCrossEntropy(kernelIdx int,
	logitsOff, targetsOff, lossesOff, gradOff uint32,
	nPos, vocabSize uint32, invN float32) {
	d.cmd("crossentropy %d %d %d %f",
		kernelIdx, nPos, vocabSize, invN)
}
