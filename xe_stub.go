//go:build !linux || !cgo

package xe

import "unsafe"

type Xe struct{}

func NewXe() *Xe                                              { return nil }
func (x *Xe) Name() string                                   { return "xe/unavailable" }
func (x *Xe) MatMul(a, b []float32, m, k, n int) []float32   { return nil }
func (x *Xe) RMSNorm(data, weight []float32, eps float32)     {}
func (x *Xe) SoftMax(data []float32, n int)                   {}
func (x *Xe) ReLU(data []float32)                             {}
func (x *Xe) VRAM() uint64                                    { return 0 }
func (x *Xe) Benchmark() float64                              { return 0 }
func (x *Xe) Sync()                                           {}
func SharedAlloc(bytes int) unsafe.Pointer                    { return nil }
func Free(ptr unsafe.Pointer)                                 {}
func Initialized() bool                                       { return false }
func LoadSPIRV(spirvData []byte, kernelName string) int       { return -1 }
func DispatchRMSNorm(shaderIdx int, x, out, weight unsafe.Pointer, dim, seqLen int, eps float32) {}
func DispatchSiLUGate(shaderIdx int, gate, up, out unsafe.Pointer, n int) {}
func DispatchAddInPlace(shaderIdx int, a, b unsafe.Pointer, n int) {}
func SyncAll()                                                {}
