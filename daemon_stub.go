//go:build !linux

package xe

import "unsafe"

const ArenaXeStart = 0

type Daemon struct{}

func NewDaemon() *Daemon                    { return nil }
func (d *Daemon) Close()                    {}
func (d *Daemon) Name() string              { return "xe/unavailable" }
func (d *Daemon) VRAM() uint64              { return 0 }
func (d *Daemon) LoadSPIRV(path, ep string) int { return -1 }
func (d *Daemon) Alloc(bytes int) unsafe.Pointer { return nil }
func (d *Daemon) Free(ptr unsafe.Pointer)    {}
func (d *Daemon) DispatchRMSNorm(ki int, x, out, w unsafe.Pointer, dim, seq int, eps float32) {}
func (d *Daemon) DispatchSiLU(ki int, gate, up, out unsafe.Pointer, n int)                    {}
func (d *Daemon) DispatchAdd(ki int, a, b unsafe.Pointer, n int)                              {}
func (d *Daemon) Sync()                     {}

type L3Bridge struct {
	Ptr  unsafe.Pointer
	Size int
}

func (b *L3Bridge) L3Float32(byteOffset, count int) []float32 { return nil }
func (b *L3Bridge) L3Uint16(byteOffset, count int) []uint16   { return nil }
func (d *Daemon) GetL3Bridge() *L3Bridge                      { return nil }
func (d *Daemon) ConvertFP32ToBF16(kernelIdx int, fp32Offset, bf16Offset, n int) error { return nil }
func (d *Daemon) HasArena() bool                               { return false }
func (d *Daemon) GoRegion(off, n int) []float32                { return nil }
func (d *Daemon) GoRegionInt32(off, n int) []int32             { return nil }
func (d *Daemon) XeRegion(off, n int) []float32                { return nil }

type CEBuffers struct {
	Logits, Targets, Losses, Grad unsafe.Pointer
}

func (b *CEBuffers) CELogitsFloat32(n int) []float32  { return nil }
func (b *CEBuffers) CETargetsInt32(n int) []int32     { return nil }
func (b *CEBuffers) CELossesFloat32(n int) []float32  { return nil }
func (b *CEBuffers) CEGradFloat32(n int) []float32    { return nil }
func (d *Daemon) AllocCEBuffers(nPos, vocabSize int) *CEBuffers { return nil }
func (d *Daemon) DispatchCrossEntropy(ki int, lo, to, oo, go2 uint32, np, vs uint32, in2 float32) {}
