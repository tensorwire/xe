// xe-daemon: persistent C process that owns the Intel Xe GPU via Level Zero.
// Go communicates via shared memory command ring + Unix socket for control.
// Exists because Intel's IGC JIT compiler crashes in Go's address space (~50%)
// but works 100% in standalone C binaries. This daemon IS the C binary.
//
// Protocol:
//   1. Go starts xe-daemon as a child process
//   2. Daemon creates L0 context, allocates shared memory, listens on Unix socket
//   3. Go mmaps the shared memory region (GPU-accessible from both processes)
//   4. Go sends commands via socket: "dispatch rmsnorm <args>" / "alloc <size>" / "sync"
//   5. Daemon executes on Xe GPU, Go reads results from shared memory
//
// Build: gcc -O2 -o xe-daemon xe-daemon.c -lze_loader -lm -lpthread
// Usage: xe-daemon /tmp/xe.sock [arena_fd]

#include <level_zero/ze_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>

// L0 globals
static ze_driver_handle_t  g_driver  = NULL;
static ze_device_handle_t  g_device  = NULL;
static ze_context_handle_t g_context = NULL;
static ze_command_list_handle_t g_cmdlist = NULL;
static char g_device_name[256] = {0};
static uint64_t g_mem_size = 0;

// SPIR-V kernel handles
#define MAX_KERNELS 16
static ze_module_handle_t g_modules[MAX_KERNELS];
static ze_kernel_handle_t g_kernels[MAX_KERNELS];
static char g_kernel_names[MAX_KERNELS][64];
static int g_kernel_count = 0;

// Shared memory allocations tracked for cleanup
#define MAX_ALLOCS 256
static void* g_allocs[MAX_ALLOCS];
static size_t g_alloc_sizes[MAX_ALLOCS];
static int g_alloc_count = 0;

// Split shared memory arena — zero-copy data exchange with Go.
// Layout:
//   [0 .. HALF)           — GO REGION: Go writes logits + targets
//   [HALF .. HALF+4096)   — GUARD: 4KB dead zone (never touched)
//   [HALF+4096 .. SIZE)   — XE REGION: Xe writes losses + gradients
//
// Go passes the memfd file descriptor as argv[2]. Both processes mmap it.
// No filesystem, no POSIX shm, no names. Just an anonymous fd + two mmaps.
#define ARENA_SIZE      (256 * 1024 * 1024)
#define ARENA_HALF      (ARENA_SIZE / 2)
#define ARENA_GUARD     4096
#define ARENA_XE_START  (ARENA_HALF + ARENA_GUARD)

static int g_arena_fd = -1;
static void* g_arena_base = NULL;

static volatile int g_running = 1;

static void sighandler(int sig) {
    g_running = 0;
}

static int init_level_zero() {
    ze_result_t r = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (r != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "[xe-daemon] zeInit failed: %d\n", r);
        return -1;
    }

    uint32_t driverCount = 0;
    zeDriverGet(&driverCount, NULL);
    if (driverCount == 0) return -2;

    ze_driver_handle_t drivers[4];
    zeDriverGet(&driverCount, drivers);

    for (uint32_t d = 0; d < driverCount; d++) {
        uint32_t devCount = 0;
        zeDeviceGet(drivers[d], &devCount, NULL);
        if (devCount == 0) continue;

        ze_device_handle_t devices[8];
        zeDeviceGet(drivers[d], &devCount, devices);

        for (uint32_t i = 0; i < devCount; i++) {
            ze_device_properties_t props = {.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
            zeDeviceGetProperties(devices[i], &props);

            if (props.type == ZE_DEVICE_TYPE_GPU && props.vendorId == 0x8086) {
                g_driver = drivers[d];
                g_device = devices[i];
                strncpy(g_device_name, props.name, 255);
                g_mem_size = 0;

                ze_device_memory_properties_t memProps[4];
                uint32_t memCount = 4;
                zeDeviceGetMemoryProperties(g_device, &memCount, memProps);
                for (uint32_t m = 0; m < memCount; m++)
                    g_mem_size += memProps[m].totalSize;

                goto found;
            }
        }
    }
    return -3;

found:;
    ze_context_desc_t ctxDesc = {.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC};
    r = zeContextCreate(g_driver, &ctxDesc, &g_context);
    if (r != ZE_RESULT_SUCCESS) return -4;

    // Find compute queue ordinal
    uint32_t qgCount = 0;
    zeDeviceGetCommandQueueGroupProperties(g_device, &qgCount, NULL);
    ze_command_queue_group_properties_t qgProps[8];
    for (uint32_t i = 0; i < qgCount && i < 8; i++)
        qgProps[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    zeDeviceGetCommandQueueGroupProperties(g_device, &qgCount, qgProps);

    uint32_t computeOrd = 0;
    for (uint32_t i = 0; i < qgCount; i++) {
        if (qgProps[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            computeOrd = i;
            break;
        }
    }

    ze_command_queue_desc_t qDesc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .ordinal = computeOrd,
        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    r = zeCommandListCreateImmediate(g_context, g_device, &qDesc, &g_cmdlist);
    if (r != ZE_RESULT_SUCCESS) return -5;

    return 0;
}

static int init_arena_from_fd(int fd) {
    g_arena_fd = fd;

    g_arena_base = mmap(NULL, ARENA_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (g_arena_base == MAP_FAILED) {
        fprintf(stderr, "[xe-daemon] arena mmap failed: %s\n", strerror(errno));
        g_arena_base = NULL;
        return -1;
    }

    mprotect((char*)g_arena_base + ARENA_HALF, ARENA_GUARD, PROT_NONE);
    fprintf(stderr, "[xe-daemon] arena mapped: fd=%d, %d MB (go: 0-%dMB, guard: 4KB, xe: %dMB-%dMB)\n",
            fd, ARENA_SIZE / (1024*1024),
            ARENA_HALF / (1024*1024),
            ARENA_XE_START / (1024*1024), ARENA_SIZE / (1024*1024));
    return 0;
}

// L0 shared memory buffers for cross-entropy.
static void* g_ce_logits = NULL;
static void* g_ce_targets = NULL;
static void* g_ce_losses = NULL;
static void* g_ce_grad = NULL;
static size_t g_ce_logits_bytes = 0;

static int ensure_ce_buffers(uint32_t n_pos, uint32_t vocab_size) {
    size_t logits_bytes = (size_t)n_pos * vocab_size * sizeof(float);
    if (g_ce_logits && g_ce_logits_bytes >= logits_bytes) return 0;

    if (g_ce_logits) { zeMemFree(g_context, g_ce_logits); }
    if (g_ce_targets) { zeMemFree(g_context, g_ce_targets); }
    if (g_ce_losses) { zeMemFree(g_context, g_ce_losses); }
    if (g_ce_grad) { zeMemFree(g_context, g_ce_grad); }

    ze_device_mem_alloc_desc_t devDesc = {.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t hostDesc = {.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    zeMemAllocShared(g_context, &devDesc, &hostDesc, logits_bytes, 64, g_device, &g_ce_logits);
    zeMemAllocShared(g_context, &devDesc, &hostDesc, n_pos * sizeof(int), 64, g_device, &g_ce_targets);
    zeMemAllocShared(g_context, &devDesc, &hostDesc, n_pos * sizeof(float), 64, g_device, &g_ce_losses);
    zeMemAllocShared(g_context, &devDesc, &hostDesc, logits_bytes, 64, g_device, &g_ce_grad);
    g_ce_logits_bytes = logits_bytes;

    if (!g_ce_logits || !g_ce_targets || !g_ce_losses || !g_ce_grad) {
        fprintf(stderr, "[xe-daemon] CE buffer alloc failed\n");
        return -1;
    }
    fprintf(stderr, "[xe-daemon] CE L0 shared: logits=%p targets=%p losses=%p grad=%p (%.1f MB)\n",
            g_ce_logits, g_ce_targets, g_ce_losses, g_ce_grad,
            (float)(logits_bytes*2 + n_pos*8) / (1024*1024));
    return 0;
}

static int dispatch_cross_entropy_direct(int kidx,
    uint32_t n_pos, uint32_t vocab_size, float inv_n) {
    if (kidx < 0 || kidx >= g_kernel_count) return -1;
    if (!g_ce_logits) return -2;

    ze_kernel_handle_t k = g_kernels[kidx];
    zeKernelSetGroupSize(k, 256, 1, 1);
    zeKernelSetArgumentValue(k, 0, sizeof(void*), &g_ce_logits);
    zeKernelSetArgumentValue(k, 1, sizeof(void*), &g_ce_targets);
    zeKernelSetArgumentValue(k, 2, sizeof(void*), &g_ce_losses);
    zeKernelSetArgumentValue(k, 3, sizeof(void*), &g_ce_grad);
    zeKernelSetArgumentValue(k, 4, sizeof(uint32_t), &vocab_size);
    zeKernelSetArgumentValue(k, 5, sizeof(float), &inv_n);

    ze_group_count_t disp = {n_pos, 1, 1};
    ze_result_t r = zeCommandListAppendLaunchKernel(g_cmdlist, k, &disp, NULL, 0, NULL);
    return (int)r;
}

// === L3 Cache Bridge ===
#define L3_BRIDGE_SIZE (64 * 1024 * 1024)

static void* g_l3_bridge = NULL;
static size_t g_l3_bridge_size = 0;

static int init_l3_bridge() {
    ze_host_mem_alloc_desc_t hostDesc = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        .flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED
    };
    ze_result_t r = zeMemAllocHost(g_context, &hostDesc, L3_BRIDGE_SIZE, 64, &g_l3_bridge);
    if (r != ZE_RESULT_SUCCESS || !g_l3_bridge) {
        fprintf(stderr, "[xe-daemon] L3 bridge alloc failed: %d\n", r);
        return -1;
    }
    g_l3_bridge_size = L3_BRIDGE_SIZE;
    memset(g_l3_bridge, 0, L3_BRIDGE_SIZE);
    fprintf(stderr, "[xe-daemon] L3 bridge: %p (%zu MB) — cache-coherent host memory\n",
            g_l3_bridge, L3_BRIDGE_SIZE / (1024*1024));
    return 0;
}

static void* xe_alloc(size_t bytes) {
    void* ptr = NULL;
    ze_device_mem_alloc_desc_t devDesc = {.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};
    ze_host_mem_alloc_desc_t hostDesc = {.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
    ze_result_t r = zeMemAllocShared(g_context, &devDesc, &hostDesc, bytes, 64, g_device, &ptr);
    if (r != ZE_RESULT_SUCCESS) return NULL;

    if (g_alloc_count < MAX_ALLOCS) {
        g_allocs[g_alloc_count] = ptr;
        g_alloc_sizes[g_alloc_count] = bytes;
        g_alloc_count++;
    }
    return ptr;
}

static void xe_free_ptr(void* ptr) {
    zeMemFree(g_context, ptr);
    for (int i = 0; i < g_alloc_count; i++) {
        if (g_allocs[i] == ptr) {
            g_allocs[i] = g_allocs[g_alloc_count - 1];
            g_alloc_sizes[i] = g_alloc_sizes[g_alloc_count - 1];
            g_alloc_count--;
            break;
        }
    }
}

static int load_spirv(const char* path, const char* entry) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = malloc(sz);
    fread(buf, 1, sz, f);
    fclose(f);

    ze_module_desc_t modDesc = {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .format = ZE_MODULE_FORMAT_IL_SPIRV,
        .inputSize = sz,
        .pInputModule = buf
    };
    ze_module_handle_t mod = NULL;
    ze_module_build_log_handle_t buildLog = NULL;
    ze_result_t r = zeModuleCreate(g_context, g_device, &modDesc, &mod, &buildLog);
    free(buf);

    if (r != ZE_RESULT_SUCCESS) {
        if (buildLog) {
            size_t logSz = 0;
            zeModuleBuildLogGetString(buildLog, &logSz, NULL);
            char* log = malloc(logSz + 1);
            zeModuleBuildLogGetString(buildLog, &logSz, log);
            fprintf(stderr, "[xe-daemon] SPIR-V build: %s\n", log);
            free(log);
            zeModuleBuildLogDestroy(buildLog);
        }
        return -2;
    }
    if (buildLog) zeModuleBuildLogDestroy(buildLog);

    ze_kernel_desc_t kDesc = {.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC, .pKernelName = entry};
    ze_kernel_handle_t kern = NULL;
    r = zeKernelCreate(mod, &kDesc, &kern);
    if (r != ZE_RESULT_SUCCESS) {
        zeModuleDestroy(mod);
        return -3;
    }

    int idx = g_kernel_count++;
    g_modules[idx] = mod;
    g_kernels[idx] = kern;
    strncpy(g_kernel_names[idx], entry, 63);
    return idx;
}

static void xe_sync() {
    zeCommandListHostSynchronize(g_cmdlist, UINT64_MAX);
}

static int dispatch_rmsnorm(int kidx, void* x, void* out, void* weight,
                             uint32_t dim, uint32_t seqLen, float eps) {
    if (kidx < 0 || kidx >= g_kernel_count) return -1;
    ze_kernel_handle_t k = g_kernels[kidx];
    zeKernelSetGroupSize(k, 256, 1, 1);
    zeKernelSetArgumentValue(k, 0, sizeof(void*), &x);
    zeKernelSetArgumentValue(k, 1, sizeof(void*), &out);
    zeKernelSetArgumentValue(k, 2, sizeof(void*), &weight);
    zeKernelSetArgumentValue(k, 3, sizeof(uint32_t), &dim);
    zeKernelSetArgumentValue(k, 4, sizeof(uint32_t), &seqLen);
    zeKernelSetArgumentValue(k, 5, sizeof(float), &eps);
    ze_group_count_t disp = {seqLen, 1, 1};
    zeCommandListAppendLaunchKernel(g_cmdlist, k, &disp, NULL, 0, NULL);
    return 0;
}

static int dispatch_silu(int kidx, void* gate, void* up, void* out, uint32_t n) {
    if (kidx < 0 || kidx >= g_kernel_count) return -1;
    ze_kernel_handle_t k = g_kernels[kidx];
    zeKernelSetGroupSize(k, 256, 1, 1);
    zeKernelSetArgumentValue(k, 0, sizeof(void*), &gate);
    zeKernelSetArgumentValue(k, 1, sizeof(void*), &up);
    zeKernelSetArgumentValue(k, 2, sizeof(void*), &out);
    zeKernelSetArgumentValue(k, 3, sizeof(uint32_t), &n);
    uint32_t groups = (n + 255) / 256;
    ze_group_count_t disp = {groups, 1, 1};
    zeCommandListAppendLaunchKernel(g_cmdlist, k, &disp, NULL, 0, NULL);
    return 0;
}

static int dispatch_add(int kidx, void* a, void* b, uint32_t n) {
    if (kidx < 0 || kidx >= g_kernel_count) return -1;
    ze_kernel_handle_t k = g_kernels[kidx];
    zeKernelSetGroupSize(k, 256, 1, 1);
    zeKernelSetArgumentValue(k, 0, sizeof(void*), &a);
    zeKernelSetArgumentValue(k, 1, sizeof(void*), &b);
    zeKernelSetArgumentValue(k, 2, sizeof(uint32_t), &n);
    uint32_t groups = (n + 255) / 256;
    ze_group_count_t disp = {groups, 1, 1};
    zeCommandListAppendLaunchKernel(g_cmdlist, k, &disp, NULL, 0, NULL);
    return 0;
}

static void handle_cmd(int fd, char* cmd) {
    char resp[512];

    if (strncmp(cmd, "info", 4) == 0) {
        snprintf(resp, sizeof(resp), "OK %s %lu %d\n",
                 g_device_name, (unsigned long)g_mem_size, g_kernel_count);
        write(fd, resp, strlen(resp));

    } else if (strncmp(cmd, "load ", 5) == 0) {
        char path[256], entry[64];
        if (sscanf(cmd + 5, "%255s %63s", path, entry) == 2) {
            int idx = load_spirv(path, entry);
            snprintf(resp, sizeof(resp), "OK %d\n", idx);
        } else {
            snprintf(resp, sizeof(resp), "ERR bad args\n");
        }
        write(fd, resp, strlen(resp));

    } else if (strncmp(cmd, "alloc ", 6) == 0) {
        size_t bytes = 0;
        sscanf(cmd + 6, "%zu", &bytes);
        void* ptr = xe_alloc(bytes);
        snprintf(resp, sizeof(resp), "OK %p\n", ptr);
        write(fd, resp, strlen(resp));

    } else if (strncmp(cmd, "free ", 5) == 0) {
        void* ptr = NULL;
        sscanf(cmd + 5, "%p", &ptr);
        if (ptr) xe_free_ptr(ptr);
        write(fd, "OK\n", 3);

    } else if (strncmp(cmd, "rmsnorm ", 8) == 0) {
        int kidx;
        void *x, *out, *w;
        uint32_t dim, seqLen;
        float eps;
        sscanf(cmd + 8, "%d %p %p %p %u %u %f", &kidx, &x, &out, &w, &dim, &seqLen, &eps);
        dispatch_rmsnorm(kidx, x, out, w, dim, seqLen, eps);
        write(fd, "OK\n", 3);

    } else if (strncmp(cmd, "silu ", 5) == 0) {
        int kidx;
        void *gate, *up, *out;
        uint32_t n;
        sscanf(cmd + 5, "%d %p %p %p %u", &kidx, &gate, &up, &out, &n);
        dispatch_silu(kidx, gate, up, out, n);
        write(fd, "OK\n", 3);

    } else if (strncmp(cmd, "add ", 4) == 0) {
        int kidx;
        void *a, *b;
        uint32_t n;
        sscanf(cmd + 4, "%d %p %p %u", &kidx, &a, &b, &n);
        dispatch_add(kidx, a, b, n);
        write(fd, "OK\n", 3);

    } else if (strncmp(cmd, "l3bridge", 8) == 0) {
        if (g_l3_bridge) {
            snprintf(resp, sizeof(resp), "OK %p %zu\n", g_l3_bridge, g_l3_bridge_size);
        } else {
            snprintf(resp, sizeof(resp), "ERR no bridge\n");
        }
        write(fd, resp, strlen(resp));

    } else if (strncmp(cmd, "arena", 5) == 0) {
        if (g_arena_base) {
            snprintf(resp, sizeof(resp), "OK %d %d %d\n",
                     ARENA_SIZE, ARENA_HALF, ARENA_XE_START);
        } else {
            snprintf(resp, sizeof(resp), "ERR no arena\n");
        }
        write(fd, resp, strlen(resp));

    } else if (strncmp(cmd, "cebufs ", 7) == 0) {
        uint32_t npos, vocab;
        sscanf(cmd + 7, "%u %u", &npos, &vocab);
        int ret = ensure_ce_buffers(npos, vocab);
        if (ret == 0) {
            snprintf(resp, sizeof(resp), "OK %p %p %p %p\n",
                     g_ce_logits, g_ce_targets, g_ce_losses, g_ce_grad);
        } else {
            snprintf(resp, sizeof(resp), "ERR alloc %d\n", ret);
        }
        write(fd, resp, strlen(resp));

    } else if (strncmp(cmd, "crossentropy ", 13) == 0) {
        int kidx;
        uint32_t npos, vocab;
        float inv_n;
        sscanf(cmd + 13, "%d %u %u %f", &kidx, &npos, &vocab, &inv_n);
        int ret = dispatch_cross_entropy_direct(kidx, npos, vocab, inv_n);
        if (ret == 0) {
            write(fd, "OK\n", 3);
        } else {
            snprintf(resp, sizeof(resp), "ERR dispatch %d\n", ret);
            write(fd, resp, strlen(resp));
        }

    } else if (strncmp(cmd, "convert_bf16 ", 13) == 0) {
        int kidx;
        uint32_t fp32_off, bf16_off, n;
        sscanf(cmd + 13, "%d %u %u %u", &kidx, &fp32_off, &bf16_off, &n);
        if (kidx < 0 || kidx >= g_kernel_count || !g_l3_bridge) {
            write(fd, "ERR bad args\n", 13);
        } else {
            void* fp32_ptr = (char*)g_l3_bridge + fp32_off;
            void* bf16_ptr = (char*)g_l3_bridge + bf16_off;
            ze_kernel_handle_t k = g_kernels[kidx];
            zeKernelSetGroupSize(k, 256, 1, 1);
            zeKernelSetArgumentValue(k, 0, sizeof(void*), &fp32_ptr);
            zeKernelSetArgumentValue(k, 1, sizeof(void*), &bf16_ptr);
            zeKernelSetArgumentValue(k, 2, sizeof(uint32_t), &n);
            ze_group_count_t disp = {(n + 255) / 256, 1, 1};
            ze_result_t r = zeCommandListAppendLaunchKernel(g_cmdlist, k, &disp, NULL, 0, NULL);
            if (r == ZE_RESULT_SUCCESS) {
                write(fd, "OK\n", 3);
            } else {
                snprintf(resp, sizeof(resp), "ERR dispatch %d\n", (int)r);
                write(fd, resp, strlen(resp));
            }
        }

    } else if (strncmp(cmd, "sync", 4) == 0) {
        xe_sync();
        write(fd, "OK\n", 3);

    } else if (strncmp(cmd, "quit", 4) == 0) {
        write(fd, "OK\n", 3);
        g_running = 0;

    } else {
        snprintf(resp, sizeof(resp), "ERR unknown: %s\n", cmd);
        write(fd, resp, strlen(resp));
    }
}

int main(int argc, char** argv) {
    const char* sock_path = argc > 1 ? argv[1] : "/tmp/xe.sock";

    signal(SIGINT, sighandler);
    signal(SIGTERM, sighandler);

    int ret = init_level_zero();
    if (ret != 0) {
        fprintf(stderr, "[xe-daemon] Level Zero init failed: %d\n", ret);
        return 1;
    }
    fprintf(stderr, "[xe-daemon] %s (%lu MB shared)\n",
            g_device_name, (unsigned long)g_mem_size / 1024 / 1024);

    if (init_l3_bridge() != 0) {
        fprintf(stderr, "[xe-daemon] WARNING: L3 bridge init failed\n");
    }

    if (argc > 2) {
        int arena_fd = atoi(argv[2]);
        if (arena_fd > 0 && init_arena_from_fd(arena_fd) != 0) {
            fprintf(stderr, "[xe-daemon] WARNING: arena init failed\n");
        }
    }

    unlink(sock_path);
    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un addr = {.sun_family = AF_UNIX};
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
    bind(srv, (struct sockaddr*)&addr, sizeof(addr));
    listen(srv, 1);

    printf("READY %s %lu\n", g_device_name, (unsigned long)g_mem_size);
    fflush(stdout);

    fprintf(stderr, "[xe-daemon] listening on %s\n", sock_path);

    while (g_running) {
        int client = accept(srv, NULL, NULL);
        if (client < 0) {
            if (errno == EINTR) continue;
            break;
        }

        char buf[4096];
        int pos = 0;
        while (g_running) {
            int n = read(client, buf + pos, sizeof(buf) - pos - 1);
            if (n <= 0) break;
            pos += n;
            buf[pos] = 0;

            char* line;
            while ((line = strchr(buf, '\n')) != NULL) {
                *line = 0;
                handle_cmd(client, buf);
                int remaining = pos - (line - buf + 1);
                memmove(buf, line + 1, remaining);
                pos = remaining;
                buf[pos] = 0;
            }
        }
        close(client);
    }

    // Cleanup
    for (int i = 0; i < g_alloc_count; i++)
        zeMemFree(g_context, g_allocs[i]);
    for (int i = 0; i < g_kernel_count; i++) {
        zeKernelDestroy(g_kernels[i]);
        zeModuleDestroy(g_modules[i]);
    }
    if (g_cmdlist) zeCommandListDestroy(g_cmdlist);
    if (g_context) zeContextDestroy(g_context);
    if (g_l3_bridge) { zeMemFree(g_context, g_l3_bridge); g_l3_bridge = NULL; }
    if (g_arena_base && g_arena_base != MAP_FAILED) {
        munmap(g_arena_base, ARENA_SIZE);
    }

    unlink(sock_path);
    fprintf(stderr, "[xe-daemon] shutdown\n");
    return 0;
}
