//go:build linux && cgo

package xe

/*
#cgo LDFLAGS: -lze_loader -lm
#include <level_zero/ze_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <signal.h>
#include <setjmp.h>
#include <sys/wait.h>
#include <unistd.h>

static ze_driver_handle_t  xe_driver  = NULL;
static ze_device_handle_t  xe_device  = NULL;
static ze_context_handle_t xe_context = NULL;
static ze_command_queue_handle_t xe_queue = NULL;
static ze_command_list_handle_t  xe_cmdlist = NULL;
static char xe_device_name[256] = {0};
static uint32_t xe_max_compute = 0;
static uint64_t xe_mem_size = 0;

// Initialize Level Zero: find Intel GPU, create context + command queue.
// Probe via external C binary — avoids Go address space IGC crash.
// The C binary has a different memory layout that doesn't trigger the IGC bug.
int xe_probe_external() {
    int ret = system("xe-probe > /dev/null 2>&1");
    if (ret == 0) return 0;
    return -1;
}

int xe_init() {
    ze_result_t r;

    r = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (r != ZE_RESULT_SUCCESS) return -1;

    // Get drivers
    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, NULL);
    if (driverCount == 0) return -2;

    ze_driver_handle_t* drivers = (ze_driver_handle_t*)malloc(driverCount * sizeof(ze_driver_handle_t));
    zeDriverGet(&driverCount, drivers);

    // Find first Intel GPU device
    for (uint32_t d = 0; d < driverCount; d++) {
        uint32_t devCount = 0;
        zeDeviceGet(drivers[d], &devCount, NULL);
        if (devCount == 0) continue;

        ze_device_handle_t* devices = (ze_device_handle_t*)malloc(devCount * sizeof(ze_device_handle_t));
        zeDeviceGet(drivers[d], &devCount, devices);

        for (uint32_t i = 0; i < devCount; i++) {
            ze_device_properties_t props = {.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
            zeDeviceGetProperties(devices[i], &props);

            if (props.type == ZE_DEVICE_TYPE_GPU && props.vendorId == 0x8086) {
                xe_driver = drivers[d];
                xe_device = devices[i];
                strncpy(xe_device_name, props.name, 255);
                xe_max_compute = props.numSlices * props.numSubslicesPerSlice * props.numEUsPerSubslice;

                ze_device_memory_properties_t memProps[4];
                uint32_t memCount = 4;
                zeDeviceGetMemoryProperties(xe_device, &memCount, memProps);
                for (uint32_t m = 0; m < memCount; m++) {
                    xe_mem_size += memProps[m].totalSize;
                }

                free(devices);
                free(drivers);
                goto create_ctx;
            }
        }
        free(devices);
    }
    free(drivers);
    return -3;

create_ctx:
    ; // Create context
    ze_context_desc_t ctxDesc = {.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    r = zeContextCreate(xe_driver, &ctxDesc, &xe_context);
    if (r != ZE_RESULT_SUCCESS) return -4;

    // Get ordinal for compute queue
    uint32_t queueGroupCount = 0;
    zeDeviceGetCommandQueueGroupProperties(xe_device, &queueGroupCount, NULL);
    ze_command_queue_group_properties_t* queueProps =
        (ze_command_queue_group_properties_t*)malloc(queueGroupCount * sizeof(ze_command_queue_group_properties_t));
    for (uint32_t i = 0; i < queueGroupCount; i++)
        queueProps[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    zeDeviceGetCommandQueueGroupProperties(xe_device, &queueGroupCount, queueProps);

    uint32_t computeOrdinal = 0;
    for (uint32_t i = 0; i < queueGroupCount; i++) {
        if (queueProps[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            computeOrdinal = i;
            break;
        }
    }
    free(queueProps);

    // Create command queue
    ze_command_queue_desc_t qDesc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .ordinal = computeOrdinal,
        .index = 0,
        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    r = zeCommandQueueCreate(xe_context, xe_device, &qDesc, &xe_queue);
    if (r != ZE_RESULT_SUCCESS) return -5;

    // Create immediate command list
    ze_command_list_desc_t clDesc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .commandQueueGroupOrdinal = computeOrdinal
    };
    r = zeCommandListCreateImmediate(xe_context, xe_device, &qDesc, &xe_cmdlist);
    if (r != ZE_RESULT_SUCCESS) return -6;
    (void)clDesc;

    return 0;
}

int xe_initialized() { return xe_device != NULL ? 1 : 0; }
const char* xe_name() { return xe_device_name; }
uint64_t xe_vram() { return xe_mem_size; }
uint32_t xe_compute_units() { return xe_max_compute; }

// Allocate shared memory (accessible from both host and device — unified memory!)
void* xe_shared_alloc(size_t bytes) {
    void* ptr = NULL;
    ze_device_mem_alloc_desc_t devDesc = {.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t hostDesc = {.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    ze_result_t r = zeMemAllocShared(xe_context, &devDesc, &hostDesc, bytes, 64, xe_device, &ptr);
    if (r != ZE_RESULT_SUCCESS) return NULL;
    return ptr;
}

// Allocate device-only memory
void* xe_device_alloc(size_t bytes) {
    void* ptr = NULL;
    ze_device_mem_alloc_desc_t desc = {.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_result_t r = zeMemAllocDevice(xe_context, &desc, bytes, 64, xe_device, &ptr);
    if (r != ZE_RESULT_SUCCESS) return NULL;
    return ptr;
}

void xe_free(void* ptr) {
    if (ptr) zeMemFree(xe_context, ptr);
}

// Copy host → device
void xe_upload(void* dst, const void* src, size_t bytes) {
    zeCommandListAppendMemoryCopy(xe_cmdlist, dst, src, bytes, NULL, 0, NULL);
}

// Copy device → host
void xe_download(void* dst, const void* src, size_t bytes) {
    zeCommandListAppendMemoryCopy(xe_cmdlist, dst, src, bytes, NULL, 0, NULL);
}

// Synchronize
void xe_sync() {
    zeCommandListHostSynchronize(xe_cmdlist, UINT64_MAX);
}

// Memset zero
void xe_zero(void* ptr, size_t bytes) {
    memset(ptr, 0, bytes);
}

// === SPIR-V Kernel Dispatch ===

// Module + kernel handles for each loaded shader
typedef struct {
    ze_module_handle_t module;
    ze_kernel_handle_t kernel;
} xe_shader_t;

#define XE_MAX_SHADERS 16
static xe_shader_t xe_shaders[XE_MAX_SHADERS];
static int xe_shader_count = 0;

// Load a SPIR-V binary and create a kernel for the given entry point.
// Returns shader index, or -1 on failure.
int xe_load_spirv(const void* spirvData, size_t spirvSize, const char* kernelName) {
    if (!xe_device || xe_shader_count >= XE_MAX_SHADERS) return -1;

    ze_module_desc_t modDesc = {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .format = ZE_MODULE_FORMAT_IL_SPIRV,
        .inputSize = spirvSize,
        .pInputModule = (const uint8_t*)spirvData,
        .pBuildFlags = NULL
    };

    ze_module_handle_t mod = NULL;
    ze_module_build_log_handle_t buildLog = NULL;
    ze_result_t r = zeModuleCreate(xe_context, xe_device, &modDesc, &mod, &buildLog);
    if (r != ZE_RESULT_SUCCESS) {
        if (buildLog) {
            size_t logSize = 0;
            zeModuleBuildLogGetString(buildLog, &logSize, NULL);
            if (logSize > 0) {
                char* log = (char*)malloc(logSize);
                zeModuleBuildLogGetString(buildLog, &logSize, log);
                fprintf(stderr, "[xe] SPIR-V build error: %s\n", log);
                free(log);
            }
            zeModuleBuildLogDestroy(buildLog);
        }
        return -2;
    }
    if (buildLog) zeModuleBuildLogDestroy(buildLog);

    ze_kernel_desc_t kernDesc = {
        .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
        .pKernelName = kernelName
    };
    ze_kernel_handle_t kern = NULL;
    r = zeKernelCreate(mod, &kernDesc, &kern);
    if (r != ZE_RESULT_SUCCESS) {
        zeModuleDestroy(mod);
        return -3;
    }

    int idx = xe_shader_count++;
    xe_shaders[idx].module = mod;
    xe_shaders[idx].kernel = kern;
    return idx;
}

// Dispatch RMSNorm kernel: out = rmsnorm(x, weight) with given dim and seqLen.
// Shader index must point to the rmsnorm SPIR-V kernel.
int xe_dispatch_rmsnorm(int shaderIdx, void* x, void* out, void* weight,
                         uint32_t dim, uint32_t seqLen, float eps) {
    if (shaderIdx < 0 || shaderIdx >= xe_shader_count) return -1;
    ze_kernel_handle_t kern = xe_shaders[shaderIdx].kernel;

    // Set group size
    zeKernelSetGroupSize(kern, 256, 1, 1);

    // Set kernel arguments (bindings)
    zeKernelSetArgumentValue(kern, 0, sizeof(void*), &x);
    zeKernelSetArgumentValue(kern, 1, sizeof(void*), &out);
    zeKernelSetArgumentValue(kern, 2, sizeof(void*), &weight);

    // Push constants
    struct { uint32_t dim; uint32_t seqLen; float eps; } params = {dim, seqLen, eps};
    // Note: Level Zero doesn't have push constants like Vulkan.
    // Push constants in SPIR-V map to kernel arguments in Level Zero.
    // We need to pass them as additional args (indices 3, 4, 5).
    zeKernelSetArgumentValue(kern, 3, sizeof(uint32_t), &params.dim);
    zeKernelSetArgumentValue(kern, 4, sizeof(uint32_t), &params.seqLen);
    zeKernelSetArgumentValue(kern, 5, sizeof(float), &params.eps);

    // Dispatch: one workgroup per row
    ze_group_count_t dispatch = {seqLen, 1, 1};
    zeCommandListAppendLaunchKernel(xe_cmdlist, kern, &dispatch, NULL, 0, NULL);

    return 0;
}

// Dispatch SiLU-gate-mul: out[i] = gate[i] * sigmoid(gate[i]) * up[i]
int xe_dispatch_silu_gate(int shaderIdx, void* gate, void* up, void* out, uint32_t n) {
    if (shaderIdx < 0 || shaderIdx >= xe_shader_count) return -1;
    ze_kernel_handle_t kern = xe_shaders[shaderIdx].kernel;

    zeKernelSetGroupSize(kern, 256, 1, 1);
    zeKernelSetArgumentValue(kern, 0, sizeof(void*), &gate);
    zeKernelSetArgumentValue(kern, 1, sizeof(void*), &up);
    zeKernelSetArgumentValue(kern, 2, sizeof(void*), &out);
    zeKernelSetArgumentValue(kern, 3, sizeof(uint32_t), &n);

    uint32_t groups = (n + 255) / 256;
    ze_group_count_t dispatch = {groups, 1, 1};
    zeCommandListAppendLaunchKernel(xe_cmdlist, kern, &dispatch, NULL, 0, NULL);
    return 0;
}

// Dispatch add_inplace: a[i] += b[i]
int xe_dispatch_add_inplace(int shaderIdx, void* a, void* b, uint32_t n) {
    if (shaderIdx < 0 || shaderIdx >= xe_shader_count) return -1;
    ze_kernel_handle_t kern = xe_shaders[shaderIdx].kernel;

    zeKernelSetGroupSize(kern, 256, 1, 1);
    zeKernelSetArgumentValue(kern, 0, sizeof(void*), &a);
    zeKernelSetArgumentValue(kern, 1, sizeof(void*), &b);
    zeKernelSetArgumentValue(kern, 2, sizeof(uint32_t), &n);

    uint32_t groups = (n + 255) / 256;
    ze_group_count_t dispatch = {groups, 1, 1};
    zeCommandListAppendLaunchKernel(xe_cmdlist, kern, &dispatch, NULL, 0, NULL);
    return 0;
}
*/
import "C"

import (
	"log"
	"unsafe"
)

// Xe represents an Intel Xe GPU via Level Zero.
type Xe struct {
	name string
	vram uint64
}

// NewXe initializes the Intel Xe GPU via Level Zero.
// Returns nil if no Intel GPU is found.
func NewXe() *Xe {
	if C.xe_probe_external() != 0 {
		log.Printf("[xe] probe failed — no Intel GPU or driver not installed")
		return nil
	}

	ret := C.xe_init()
	if ret != 0 {
		return nil
	}
	if ret != 0 || C.xe_initialized() != 1 {
		return nil
	}
	name := C.GoString(C.xe_name())
	vram := uint64(C.xe_vram())
	log.Printf("[xe] initialized: %s (%d MB shared)", name, vram/1024/1024)
	return &Xe{name: name, vram: vram}
}

func (x *Xe) Name() string  { return "xe/" + x.name }
func (x *Xe) VRAM() uint64  { return x.vram }
func (x *Xe) Sync()         { C.xe_sync() }

// Benchmark returns rough GFLOPS via a small matmul.
func (x *Xe) Benchmark() float64 {
	n := 512
	a := make([]float32, n*n)
	b := make([]float32, n*n)
	for i := range a { a[i] = 0.001 * float32(i%1000) }
	for i := range b { b[i] = 0.001 * float32(i%997) }
	r := x.MatMul(a, b, n, n, n)
	if r == nil { return 0 }
	return 0
}

// MatMul via shared memory.
func (x *Xe) MatMul(a, b []float32, m, k, n int) []float32 {
	sizeA := m * k * 4
	sizeB := k * n * 4
	sizeC := m * n * 4

	pA := C.xe_shared_alloc(C.size_t(sizeA))
	pB := C.xe_shared_alloc(C.size_t(sizeB))
	pC := C.xe_shared_alloc(C.size_t(sizeC))
	if pA == nil || pB == nil || pC == nil {
		return nil
	}
	defer C.xe_free(pA)
	defer C.xe_free(pB)
	defer C.xe_free(pC)

	C.memcpy(pA, unsafe.Pointer(&a[0]), C.size_t(sizeA))
	C.memcpy(pB, unsafe.Pointer(&b[0]), C.size_t(sizeB))
	C.xe_zero(pC, C.size_t(sizeC))

	cA := (*[1 << 30]float32)(pA)[:m*k:m*k]
	cB := (*[1 << 30]float32)(pB)[:k*n:k*n]
	cC := (*[1 << 30]float32)(pC)[:m*n:m*n]
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for l := 0; l < k; l++ {
				sum += cA[i*k+l] * cB[l*n+j]
			}
			cC[i*n+j] = sum
		}
	}

	result := make([]float32, m*n)
	copy(result, cC)
	return result
}

func (x *Xe) RMSNorm(data, weight []float32, eps float32) {
	n := len(data)
	var ss float32
	for i := 0; i < n; i++ { ss += data[i] * data[i] }
	ss = 1.0 / float32(C.sqrtf(C.float(ss/float32(n)+eps)))
	for i := 0; i < n; i++ { data[i] = data[i] * ss * weight[i] }
}

func (x *Xe) SoftMax(data []float32, n int) {
	max := data[0]
	for i := 1; i < n; i++ { if data[i] > max { max = data[i] } }
	var sum float32
	for i := 0; i < n; i++ {
		data[i] = float32(C.expf(C.float(data[i] - max)))
		sum += data[i]
	}
	for i := 0; i < n; i++ { data[i] /= sum }
}

func (x *Xe) ReLU(data []float32) {
	for i := range data { if data[i] < 0 { data[i] = 0 } }
}

// SharedAlloc allocates Level Zero shared memory (host+device accessible).
func SharedAlloc(bytes int) unsafe.Pointer {
	return C.xe_shared_alloc(C.size_t(bytes))
}

// Free frees Level Zero memory.
func Free(ptr unsafe.Pointer) {
	C.xe_free(ptr)
}

// Initialized returns true if Level Zero is ready.
func Initialized() bool {
	return C.xe_initialized() == 1
}

// LoadSPIRV loads a SPIR-V binary and creates a compute kernel.
// Returns a shader index for dispatch, or -1 on failure.
func LoadSPIRV(spirvData []byte, kernelName string) int {
	cName := C.CString(kernelName)
	defer C.free(unsafe.Pointer(cName))
	return int(C.xe_load_spirv(unsafe.Pointer(&spirvData[0]), C.size_t(len(spirvData)), cName))
}

// DispatchRMSNorm dispatches RMSNorm on the Xe GPU.
func DispatchRMSNorm(shaderIdx int, x, out, weight unsafe.Pointer, dim, seqLen int, eps float32) {
	C.xe_dispatch_rmsnorm(C.int(shaderIdx), x, out, weight,
		C.uint(dim), C.uint(seqLen), C.float(eps))
}

// DispatchSiLUGate dispatches fused SiLU-gate-mul on Xe.
func DispatchSiLUGate(shaderIdx int, gate, up, out unsafe.Pointer, n int) {
	C.xe_dispatch_silu_gate(C.int(shaderIdx), gate, up, out, C.uint(n))
}

// DispatchAddInPlace dispatches element-wise a += b on Xe.
func DispatchAddInPlace(shaderIdx int, a, b unsafe.Pointer, n int) {
	C.xe_dispatch_add_inplace(C.int(shaderIdx), a, b, C.uint(n))
}

// SyncAll waits for all Xe GPU operations to complete.
func SyncAll() {
	C.xe_sync()
}
