#pragma GCC diagnostic ignored "-Woverlength-strings"
#ifdef __clang__
#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
#endif

#include "ggml-rknn.h"
#include "ggml-backend.h"
#include "ggml-impl.h" // i
#include "ggml-backend-impl.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "fp16/Float16.h"
#include "model_related_config.h"

#include <string.h>
#include <cstring>

#include <thread>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <regex>
#include <mutex>

#include <cstdarg>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <atomic>
#include <fstream>
#include <limits>
#include <string>
#include <cmath>
#include <iostream>
#include <fcntl.h>
#include <sys/sysinfo.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <arm_neon.h>
#include <json.hpp>
using json = nlohmann::json;

// #define RKNN_MATMUL_DEBUG_TIMING_INFO
#define GGML_COMMON_DECL_C
// #define RKNN_MATMUL_DEBUG

// #define RKNN_MATMUL_DEBUG_TIMING_INFO

// #define RKNN_MATMUL_DEBUG_TIMING_DETAILS

// #define RKNN_NPU_CORE_DEBUG

#define GGML_RKNPU2_USE_OUTSIDE_ALLOC 0

#include "ggml-common.h"

using namespace rknpu2;
#define UNUSED(x) (void)(x)

#define GGML_RKNPU2_INPUT_SCALE 1.7f

#define GGML_NAME_MAX 64

#define MAX_RKNN_MEMORY (1024ULL * 1024 * 1024 * 4) // 4GB

#define MAX_K 8192 // hardware flaw
#define MAX_M_WARMUP 512

//MARK: LOCAL VAR

uint32_t omp_threads = 4;

uint64_t rknpu2_allocated_bytes = 0;

// Externally-set prefill/decode phase.
// Set to true before a prefill step, false before a decode step, via
// ggml_backend_rknn_set_is_prefill().  Used only in ggml_rk_compute_forward
// for runtime behaviour (logging, future per-phase logic).
// supports_op always uses ne1>1 which is reliable during sched_reserve.
static std::atomic<bool> g_rknn_is_prefill{false};
static std::atomic<bool> g_rknn_prefill_explicitly_set{false};

// Warmup flag. Set to true via ggml_backend_rknn_set_warmup(true) before the
// warmup decode, and false after.  During warmup, supports_op checks BOTH
// prefill and decode pattern lists (union) so that all kernels are initialized.
static std::atomic<bool> g_rknn_is_warmup{false};

// Hashset for fast loaded_nodes lookup (includes nodes dynamically added during warmup)
static std::unordered_set<std::string> loaded_nodes_set;

// User-configured decode whitelist (only nodes from rknn-config.json "loaded_nodes").
// Unlike loaded_nodes_set, this is NEVER modified at runtime — it is the ground truth
// for what should be offloaded in the decode phase.
static std::unordered_set<std::string> user_loaded_nodes_set;

// Pre-compiled regex patterns for offload_nodes (decode, compiled once at config load)
static std::vector<std::regex> compiled_offload_patterns;

// Cache for decode offload node match results (node_name -> matches_any_pattern)
static std::unordered_map<std::string, bool> offload_match_cache;

// Pre-compiled regex patterns for prefill_offload_nodes (compiled once at config load)
static std::vector<std::regex> compiled_prefill_offload_patterns;

// Cache for prefill offload node match results (node_name -> matches_any_pattern)
static std::unordered_map<std::string, bool> prefill_offload_match_cache;

// --- Post-warmup frozen offload sets ---
// After warmup completes, the match caches are frozen into plain hash-sets.
// supports_op uses these for O(1) lookup, completely bypassing regex/cache logic.
static bool g_warmup_done = false;
static bool g_prefill_release_done = false;
static std::unordered_set<std::string> frozen_decode_offload_set;   // op names that matched decode patterns
static std::unordered_set<std::string> frozen_prefill_offload_set;  // op names that matched prefill patterns

// Per-operation NPU core mask: list of (compiled_pattern, core_mask_bits) pairs
// core_mask_bits is a bitmask: bit0=CORE_0, bit1=CORE_1, bit2=CORE_2 (e.g. 0x6 = CORE_1+CORE_2)
static std::vector<std::pair<std::regex, int>> compiled_op_npu_core_patterns;

// Cache for per-op NPU core mask lookups (node_name -> core_mask_bits)
static std::unordered_map<std::string, int> op_npu_core_cache;

// Pre-compiled regex patterns for ac_layout_perf_nodes (decode phase: matching nodes get ac_layout_perf=true)
static std::vector<std::regex> compiled_ac_layout_perf_patterns;
static std::unordered_map<std::string, bool> ac_layout_perf_match_cache;

// Pre-compiled regex patterns for ac_layout_perf_nodes_prefill (prefill phase).
static std::vector<std::regex> compiled_ac_layout_perf_prefill_patterns;
static std::unordered_map<std::string, bool> ac_layout_perf_prefill_match_cache;



//MARK: TIMING DEBUG

#ifdef RKNN_MATMUL_DEBUG_TIMING_INFO
    // Timing macro - usage: TIMEIT(expression, &time_accumulator_variable)
    // better just one function inside, instead of block of statements for readability
    #define TIMEIT(expr, time_accumulator) \
        do { \
            auto __timeit_start = std::chrono::high_resolution_clock::now(); \
            expr; \
            auto __timeit_end = std::chrono::high_resolution_clock::now(); \
            auto __timeit_duration = std::chrono::duration_cast<std::chrono::microseconds>(__timeit_end - __timeit_start); \
            *(time_accumulator) += __timeit_duration.count(); \
        } while(0)
#else
    #define TIMEIT(expr, time_accumulator) \
        do { \
            expr; \
        } while(0)
#endif


#ifdef RKNN_MATMUL_DEBUG_TIMING_DETAILS
    #define timing_debug_printf(...) printf(__VA_ARGS__)
#else
    #define timing_debug_printf(...) (0)
#endif

// Uncomment (or pass -DRKNN_NPU_CORE_DEBUG) to enable NPU core-assignment debug prints
// #define RKNN_NPU_CORE_DEBUG

#ifdef RKNN_NPU_CORE_DEBUG
    #define npu_core_debug_printf(...) printf(__VA_ARGS__)
#else
    #define npu_core_debug_printf(...) (0)
#endif

// Helper: convert rknn_core_mask bitmask to human-readable string
static inline const char* core_mask_to_str(int mask) {
    switch (mask) {
        case 1: return "CORE_0";
        case 2: return "CORE_1";
        case 4: return "CORE_2";
        case 3: return "CORE_0_1";
        case 5: return "CORE_0_2";
        case 6: return "CORE_1_2";
        case 7: return "CORE_0_1_2";
        default: return "CORE_AUTO";
    }
}

// Helper: given a core_mask_bits bitmask (e.g. 0x6 = bit1+bit2 = CORE_1+CORE_2),
// returns the single-core mask (1<<bitN) assigned to thread_idx.
// Bits are enumerated in ascending order (CORE_0 first).
// If thread_idx >= popcount(core_mask_bits), wraps around.
static inline int npu_core_for_thread(int core_mask_bits, int thread_idx) {
    int count = 0;
    int first_bit_mask = 0;
    for (int bit = 0; bit < 8; bit++) {
        if (core_mask_bits & (1 << bit)) {
            if (first_bit_mask == 0) first_bit_mask = (1 << bit);
            if (count == thread_idx) return (1 << bit);
            count++;
        }
    }
    // Wrap around (thread_idx >= popcount) — assign to first available core
    return first_bit_mask ? first_bit_mask : 1;
}

#if GGML_RKNPU2_USE_OUTSIDE_ALLOC
//MARK: DMA
struct dma_heap_allocation_data {
	uint64_t len;
	uint32_t fd;
	uint32_t fd_flags;
	uint64_t heap_flags;
};

#define DMA_HEAP_IOC_MAGIC		'H'
#define DMA_HEAP_IOCTL_ALLOC	_IOWR(DMA_HEAP_IOC_MAGIC, 0x0,\
				      struct dma_heap_allocation_data)

#define DMA_BUF_SYNC_READ      (1 << 0)
#define DMA_BUF_SYNC_WRITE     (2 << 0)
#define DMA_BUF_SYNC_RW        (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START     (0 << 2)
#define DMA_BUF_SYNC_END       (1 << 2)
#define DMA_BUF_BASE		'b'
#define DMA_BUF_IOCTL_SYNC	_IOW(DMA_BUF_BASE, 0, uint64_t)
#define CMA_HEAP_SIZE	(1024 * 1024)

//Helper function to manually allocate buffer from dma_heap for RKNPU2
//The internal RKNPU2 API will allocate buffer from DMA32 heap, which is only 4GiB, not enough for large models.
//WARNING: Memory leak will not be released on exit!! But it will be released on next run...?
int dma_alloc(size_t size, int *fd, void **va) {
    int ret;
    int prot;
    void *mmap_va;
    int dma_heap_fd = -1;
    struct dma_heap_allocation_data buf_data;
    const char* path = "/dev/dma_heap/system";

    /* open dma_heap fd */
    dma_heap_fd = open(path, O_RDWR);
    if (dma_heap_fd < 0) {
        printf("open %s fail!\n", path);
        return dma_heap_fd;
    }

    /* alloc buffer */
    memset(&buf_data, 0x0, sizeof(struct dma_heap_allocation_data));

    buf_data.len = size;
    buf_data.fd_flags = O_CLOEXEC | O_RDWR;
    ret = ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &buf_data);
    if (ret < 0) {
        printf("RK_DMA_HEAP_ALLOC_BUFFER failed\n");
        return ret;
    }

    /* mmap va */
    if (fcntl(buf_data.fd, F_GETFL) & O_RDWR)
        prot = PROT_READ | PROT_WRITE;
    else
        prot = PROT_READ;

    /* mmap contiguors buffer to user */
    mmap_va = (void *)mmap(NULL, buf_data.len, prot, MAP_SHARED, buf_data.fd, 0);
    if (mmap_va == MAP_FAILED) {
        printf("mmap failed: %s\n", strerror(errno));
        return -errno;
    }

    *va = mmap_va;
    *fd = buf_data.fd;

    close(dma_heap_fd);

    return 0;
}

int dma_sync_device_to_cpu(int fd) {
    uint64_t flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags);
}

int dma_sync_cpu_to_device(int fd) {
    uint64_t flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_RW;
    return ioctl(fd, DMA_BUF_IOCTL_SYNC, &flags);
}
void dma_buf_free(size_t size, int *fd, void *va) {
    int len;

    len =  size;
    munmap(va, len);

    close(*fd);
    *fd = -1;
}

#endif

//MARK: TYPES

enum matrix_t{
    FLOAT16,
    INT8,
    INT4
};


//MARK: CONFIG
// Set default config values for rknn_config
void set_default_rknn_config(json *rknn_config_ptr) {
    (*rknn_config_ptr)["npu_prefill"] = true;
    (*rknn_config_ptr)["npu_decode"] = true;
    (*rknn_config_ptr)["offload_nodes"] = json::array({"ffn.*", ".*output"});
    (*rknn_config_ptr)["prefill_offload_nodes"] = json::array({"ffn.*", ".*output"});
    (*rknn_config_ptr)["loaded_nodes"] = json::array({});
    (*rknn_config_ptr)["regex"] = true;
}

static json local_rknn_config = {
    {"npu_prefill", false},
    {"npu_decode", false}, 
    {"offload_nodes", json::array({"ffn.*", ".*output"})},
    {"prefill_offload_nodes", json::array({"ffn.*", ".*output"})},
    {"loaded_nodes", json::array({})},
    {"omp_threads", 4},
    {"regex", true},
    {"npu_core_mask", 7},  // bitmask: bit0=CORE_0 bit1=CORE_1 bit2=CORE_2, default=0x7 (all 3)
    {"op_npu_cores", json::array({})},
    {"ac_layout_perf_nodes", json::array({})},          // decode phase: matching nodes get ac_layout_perf=true
    {"ac_layout_perf_nodes_prefill", json::array({})}   // prefill phase: matching nodes get ac_layout_perf=true
};



void init_rknn_config(json *rknn_config_ptr) {
    std::string filename = std::string(CONFIG_DIR) + "/rknn-config.json";
    std::ifstream ifs(filename);

    json file_rknn_config;
    
    if (!ifs.is_open()) {
        std::cerr << "Cannot open JSON file: " << filename << std::endl;
        set_default_rknn_config(&file_rknn_config); 
    } else {
        file_rknn_config = json::parse(ifs);
    }

    // env variables have higher priority 
    // Check environment variables first
    bool has_env = false;
    const char* npu_prefill_env = getenv("NPU_PREFILL");
    if (npu_prefill_env != nullptr) {
        (*rknn_config_ptr)["npu_prefill"] = (strcmp(npu_prefill_env, "1") == 0 || 
                                      strcasecmp(npu_prefill_env, "true") == 0);
        has_env = true;
    } else {
        (*rknn_config_ptr)["npu_prefill"] = file_rknn_config["npu_prefill"];
    }
    
    const char* npu_decode_env = getenv("NPU_DECODE");
    if (npu_decode_env != nullptr) {
        (*rknn_config_ptr)["npu_decode"] = (strcmp(npu_decode_env, "1") == 0 || 
                                     strcasecmp(npu_decode_env, "true") == 0);
        has_env = true;
    } else {
        (*rknn_config_ptr)["npu_decode"] = file_rknn_config["npu_decode"];
    }

    const char* offload_nodes_env = getenv("OFFLOAD_NODES");
    if (offload_nodes_env != nullptr) {
        (*rknn_config_ptr)["offload_nodes"] = json::parse(offload_nodes_env);
        has_env = true;
    } else {
        (*rknn_config_ptr)["offload_nodes"] = file_rknn_config["offload_nodes"];
    }

    const char* prefill_offload_nodes_env = getenv("PREFILL_OFFLOAD_NODES");
    if (prefill_offload_nodes_env != nullptr) {
        (*rknn_config_ptr)["prefill_offload_nodes"] = json::parse(prefill_offload_nodes_env);
        has_env = true;
    } else if (file_rknn_config.contains("prefill_offload_nodes")) {
        (*rknn_config_ptr)["prefill_offload_nodes"] = file_rknn_config["prefill_offload_nodes"];
    } else {
        // Fall back to offload_nodes if prefill_offload_nodes is not configured
        (*rknn_config_ptr)["prefill_offload_nodes"] = (*rknn_config_ptr)["offload_nodes"];
    }

    const char* loaded_nodes_env = getenv("LOADED_NODES");
    if (loaded_nodes_env != nullptr) {
        (*rknn_config_ptr)["loaded_nodes"] = json::parse(loaded_nodes_env);
        has_env = true;
    } else {
        (*rknn_config_ptr)["loaded_nodes"] = file_rknn_config["loaded_nodes"];
    }

    // npu_core_mask: bitmask of which NPU cores to use (e.g. 7=all3, 3=CORE_0+CORE_1, 6=CORE_1+CORE_2)
    // Also accept legacy num_npu_cores (integer count) for backward compatibility: converted to mask (1<<n)-1
    const char* npu_core_mask_env = getenv("npu_core_mask");
    const char* num_npu_cores_env = getenv("num_npu_cores");  // legacy
    if (npu_core_mask_env != nullptr) {
        (*rknn_config_ptr)["npu_core_mask"] = std::stoi(npu_core_mask_env);
        has_env = true;
    } else if (num_npu_cores_env != nullptr) {
        // Convert count N to mask (1<<N)-1: e.g. N=2 -> 0x3 (CORE_0+CORE_1)
        int n = std::stoi(num_npu_cores_env);
        (*rknn_config_ptr)["npu_core_mask"] = (1 << n) - 1;
        has_env = true;
    } else if (file_rknn_config.contains("npu_core_mask")) {
        (*rknn_config_ptr)["npu_core_mask"] = file_rknn_config["npu_core_mask"];
    } else if (file_rknn_config.contains("num_npu_cores")) {
        // Legacy config field: convert to mask
        int n = file_rknn_config["num_npu_cores"].get<int>();
        (*rknn_config_ptr)["npu_core_mask"] = (1 << n) - 1;
    } else {
        (*rknn_config_ptr)["npu_core_mask"] = local_rknn_config["npu_core_mask"];
    }

    // Parse op_npu_cores: array of {"pattern": "...", "core_mask": bitmask}
    // core_mask is a bitmask of NPU cores (bit0=CORE_0, bit1=CORE_1, bit2=CORE_2).
    // Also accepts legacy "num_cores" integer field (converted to (1<<N)-1 mask).
    // Per-op override takes priority over global npu_core_mask.
    // Env var OP_NPU_CORES (JSON array) takes priority over config file.
    compiled_op_npu_core_patterns.clear();
    op_npu_core_cache.clear();
    const char* op_npu_cores_env = getenv("OP_NPU_CORES");
    if (op_npu_cores_env != nullptr) {
        (*rknn_config_ptr)["op_npu_cores"] = json::parse(op_npu_cores_env);
        has_env = true;
    } else if (file_rknn_config.contains("op_npu_cores")) {
        (*rknn_config_ptr)["op_npu_cores"] = file_rknn_config["op_npu_cores"];
    } else {
        (*rknn_config_ptr)["op_npu_cores"] = json::array({});
    }
    for (const auto &entry : (*rknn_config_ptr)["op_npu_cores"]) {
        int mask;
        if (entry.contains("core_mask")) {
            mask = entry["core_mask"].get<int>();
        } else {
            // Legacy "num_cores": N -> mask (1<<N)-1
            int n = entry["num_cores"].get<int>();
            mask = (1 << n) - 1;
        }
        compiled_op_npu_core_patterns.emplace_back(
            std::regex(entry["pattern"].get<std::string>(), std::regex::optimize),
            mask);
    }

    // Populate loaded_nodes_set and user_loaded_nodes_set from config.
    // user_loaded_nodes_set is the immutable user-configured decode whitelist.
    // loaded_nodes_set will also accumulate nodes dynamically loaded during warmup.
    loaded_nodes_set.clear();
    user_loaded_nodes_set.clear();
    for (const auto &node_name : (*rknn_config_ptr)["loaded_nodes"]) {
        const std::string s = node_name.get<std::string>();
        loaded_nodes_set.insert(s);
        user_loaded_nodes_set.insert(s);
    }

    // Reset post-warmup frozen sets (will be rebuilt after next warmup)
    g_warmup_done = false;
    frozen_decode_offload_set.clear();
    frozen_prefill_offload_set.clear();

    // Pre-compile offload_nodes regex patterns for decode (done once at config load)
    compiled_offload_patterns.clear();
    offload_match_cache.clear();
    for (const auto &node_name : (*rknn_config_ptr)["offload_nodes"]) {
        compiled_offload_patterns.emplace_back(
            node_name.get<std::string>(), std::regex::optimize);
    }

    // Pre-compile prefill_offload_nodes regex patterns (done once at config load)
    compiled_prefill_offload_patterns.clear();
    prefill_offload_match_cache.clear();
    for (const auto &node_name : (*rknn_config_ptr)["prefill_offload_nodes"]) {
        compiled_prefill_offload_patterns.emplace_back(
            node_name.get<std::string>(), std::regex::optimize);
    }

    //TODO: not used
    const char* regex_env = getenv("REGEX");
    if (regex_env != nullptr) {
        (*rknn_config_ptr)["regex"] = (strcmp(regex_env, "1") == 0 || 
                                     strcasecmp(regex_env, "true") == 0);
        has_env = true;
    } else {
        (*rknn_config_ptr)["regex"] = file_rknn_config["regex"];
    }

    const char* omp_threads_env = getenv("OMP_THREADS");
    if (omp_threads_env != nullptr) {
        (*rknn_config_ptr)["omp_threads"] = std::stoi(omp_threads_env);
        has_env = true;
    } else {
        (*rknn_config_ptr)["omp_threads"] = file_rknn_config["omp_threads"];
    }

    // ac_layout_perf_nodes: list of regex; nodes whose name matches any pattern get ac_layout_perf=true
    const char* ac_layout_perf_nodes_env = getenv("AC_LAYOUT_PERF_NODES");
    if (ac_layout_perf_nodes_env != nullptr) {
        (*rknn_config_ptr)["ac_layout_perf_nodes"] = json::parse(ac_layout_perf_nodes_env);
        has_env = true;
    } else if (file_rknn_config.contains("ac_layout_perf_nodes")) {
        (*rknn_config_ptr)["ac_layout_perf_nodes"] = file_rknn_config["ac_layout_perf_nodes"];
    } else {
        (*rknn_config_ptr)["ac_layout_perf_nodes"] = local_rknn_config["ac_layout_perf_nodes"];
    }
    compiled_ac_layout_perf_patterns.clear();
    ac_layout_perf_match_cache.clear();
    for (const auto &node_name : (*rknn_config_ptr)["ac_layout_perf_nodes"]) {
        compiled_ac_layout_perf_patterns.emplace_back(
            node_name.get<std::string>(), std::regex::optimize);
    }

    // ac_layout_perf_nodes_prefill: same structure, but for the prefill phase
    const char* ac_layout_perf_nodes_prefill_env = getenv("AC_LAYOUT_PERF_NODES_PREFILL");
    if (ac_layout_perf_nodes_prefill_env != nullptr) {
        (*rknn_config_ptr)["ac_layout_perf_nodes_prefill"] = json::parse(ac_layout_perf_nodes_prefill_env);
        has_env = true;
    } else if (file_rknn_config.contains("ac_layout_perf_nodes_prefill")) {
        (*rknn_config_ptr)["ac_layout_perf_nodes_prefill"] = file_rknn_config["ac_layout_perf_nodes_prefill"];
    } else {
        (*rknn_config_ptr)["ac_layout_perf_nodes_prefill"] = local_rknn_config["ac_layout_perf_nodes_prefill"];
    }
    compiled_ac_layout_perf_prefill_patterns.clear();
    ac_layout_perf_prefill_match_cache.clear();
    for (const auto &node_name : (*rknn_config_ptr)["ac_layout_perf_nodes_prefill"]) {
        compiled_ac_layout_perf_prefill_patterns.emplace_back(
            node_name.get<std::string>(), std::regex::optimize);
    }

    if (has_env) {
        // set_default_rknn_config(rknn_config_ptr); 
        printf("ggml-rknn: using environment variables\n");
    }
    
    printf("ggml-rknn: npu_prefill: %d\n", (*rknn_config_ptr)["npu_prefill"].get<bool>());
    printf("ggml-rknn: npu_decode: %d\n", (*rknn_config_ptr)["npu_decode"].get<bool>());
    printf("ggml-rknn: offload_nodes (decode): %s\n", (*rknn_config_ptr)["offload_nodes"].dump().c_str());
    printf("ggml-rknn: prefill_offload_nodes: %s\n", (*rknn_config_ptr)["prefill_offload_nodes"].dump().c_str());

    std::string loaded_nodes_str = "{";
    for (auto it = loaded_nodes_set.begin(); it != loaded_nodes_set.end(); ++it) {
        if (it != loaded_nodes_set.begin()) {loaded_nodes_str += ", ";}
        loaded_nodes_str += "\"" + *it + "\"";
    }
    loaded_nodes_str += "}";
    printf("ggml-rknn: loaded_nodes set: %s\n", loaded_nodes_str.c_str());

    printf("ggml-rknn: ac_layout_perf_nodes (decode): %s\n", (*rknn_config_ptr)["ac_layout_perf_nodes"].dump().c_str());
    printf("ggml-rknn: ac_layout_perf_nodes_prefill: %s\n", (*rknn_config_ptr)["ac_layout_perf_nodes_prefill"].dump().c_str());
    printf("ggml-rknn: complete config dump: %s\n", (*rknn_config_ptr).dump().c_str());
    printf("ggml-rknn: op_npu_cores: %s\n", (*rknn_config_ptr)["op_npu_cores"].dump().c_str());

}

// Returns the NPU core bitmask for the given operation name.
// Matches compiled_op_npu_core_patterns in order; returns default_mask if none match.
// Results are cached in op_npu_core_cache for fast repeated lookup.
// Bitmask: bit0=CORE_0, bit1=CORE_1, bit2=CORE_2 (e.g. 0x6 = CORE_1+CORE_2).
static int get_op_npu_core_mask(const char *name, int default_mask) {
    std::string key(name);
    auto it = op_npu_core_cache.find(key);
    if (it != op_npu_core_cache.end()) {
        npu_core_debug_printf("[NPU_CORE] get_op_npu_core_mask: \"%s\" -> 0x%x %s (cached)\n", name, it->second, core_mask_to_str(it->second));
        return it->second;
    }
    for (size_t i = 0; i < compiled_op_npu_core_patterns.size(); ++i) {
        const auto &pat_mask = compiled_op_npu_core_patterns[i];
        if (std::regex_search(key, pat_mask.first)) {
            op_npu_core_cache[key] = pat_mask.second;
            npu_core_debug_printf("[NPU_CORE] get_op_npu_core_mask: \"%s\" matched pattern #%zu -> core_mask=0x%x (%s)\n", name, i, pat_mask.second, core_mask_to_str(pat_mask.second));
            return pat_mask.second;
        }
    }
    op_npu_core_cache[key] = default_mask;
    npu_core_debug_printf("[NPU_CORE] get_op_npu_core_mask: \"%s\" no pattern matched -> default=0x%x (%s)\n", name, default_mask, core_mask_to_str(default_mask));
    return default_mask;
}

// Returns true if node name matches any ac_layout_perf_nodes regex (decode phase perf layout).
// Results are cached in ac_layout_perf_match_cache.
static bool match_ac_layout_perf_node(const char *node_name) {
    if (compiled_ac_layout_perf_patterns.empty()) {
        return false;
    }
    std::string key(node_name);
    auto it = ac_layout_perf_match_cache.find(key);
    if (it != ac_layout_perf_match_cache.end()) {
        return it->second;
    }
    bool matched = false;
    for (const auto &pattern : compiled_ac_layout_perf_patterns) {
        if (std::regex_match(key, pattern)) {
            matched = true;
            break;
        }
    }
    ac_layout_perf_match_cache[key] = matched;
    return matched;
}

// Returns true if node name matches any ac_layout_perf_nodes_prefill regex (prefill phase perf layout).
// Results are cached in ac_layout_perf_prefill_match_cache.
static bool match_ac_layout_perf_prefill_node(const char *node_name) {
    if (compiled_ac_layout_perf_prefill_patterns.empty()) {
        return false;
    }
    std::string key(node_name);
    auto it = ac_layout_perf_prefill_match_cache.find(key);
    if (it != ac_layout_perf_prefill_match_cache.end()) {
        return it->second;
    }
    bool matched = false;
    for (const auto &pattern : compiled_ac_layout_perf_prefill_patterns) {
        if (std::regex_match(key, pattern)) {
            matched = true;
            break;
        }
    }
    ac_layout_perf_prefill_match_cache[key] = matched;
    return matched;
}


inline size_t get_element_size(matrix_t type) {
    switch(type) {
        case FLOAT16: return sizeof(float16);
        case INT8:    return sizeof(int8_t);
        case INT4:    return sizeof(int8_t); // INT4 packed in int8_t
        default:      return 0;
    }
}


//MARK: PADDING HELPER
struct mat_info{
    int64_t row;
    int64_t col;
    int64_t pad_row;
    int64_t pad_col;
    matrix_t matrix_type;
    void * ori_data;
    void * pad_data;
    size_t ori_size;
    size_t pad_size;
    char * matrix_name;
    bool is_padded=false;
    bool is_A=false;

    mat_info(int64_t row, 
        int64_t col,
        int64_t pad_row, 
        int64_t pad_col, 
        matrix_t matrix_type, 
        void * ori_data, 
        void * pad_data, 
        size_t ori_size, 
        size_t pad_size, 
        char * matrix_name, 
        bool is_padded, 
        bool is_A): 
        row(row), 
        col(col), 
        pad_row(pad_row), 
        pad_col(pad_col), 
        matrix_type(matrix_type), 
        ori_data(ori_data), 
        pad_data(pad_data), 
        ori_size(ori_size), 
        pad_size(pad_size), 
        matrix_name(matrix_name), 
        is_padded(is_padded), 
        is_A(is_A)
        {}

    mat_info(int64_t row_, 
        int64_t col_,
        matrix_t matrix_type_, 
        void * origin_data_, 
        bool is_A_, 
        char* name_ = NULL)
    : mat_info(
            row_,
            col_,
            row_,
            col_,
            matrix_type_,
            origin_data_,
            origin_data_,
            row_ * col_ * get_element_size(matrix_type_),
            row_ * col_ * get_element_size(matrix_type_),
            name_,
            false,
            is_A_
        )
    {
        // notes for padding: 
        // for A (M, K) (intermediate values), 
        //     performance layout [K1, M, subK] requires K to be aligned with subK (8 for F16) 
        // for B (K, N) (weights), 
        //     native layout [N1, K, subN, subK] requires 
        //     K to be aligned with subK (32 for F16), 
        //     N aligned with subN (16 for F16)
        // So at the end K will need to be aligned with 32, and N will need to be aligned with 16

        //TODO: also here K_align N_align is hardcoded 
        if(is_A_){
            // (M, K) 
            // this->pad_col = ((col_ - 1) / 32 + 1) * 32;
            // it's hard to pad K here, since A is now row-major and we're padding the end of rows 
            GGML_ASSERT(col_ % 32 == 0);
            
        }
        else {
            // no, B is already in native layout here
        }
    }

    mat_info(){
        
    }
};

//MARK: KERNEL HELPER
struct ggml_rknpu_matmul_part_AC {
    int M;
    int K;
    int N;
    rknn_matmul_type type;
    rknn_tensor_mem* A;
    rknn_tensor_mem* C;

    // considering prefill, AC's io_attr has higher priority than B's
    rknn_matmul_io_attr io_attr;

    // default is false, which means normal layout (row-major)
    bool ac_layout_perf = false;

    rknn_matmul_ctx ctx = 0;
    bool prefill = false;
    int thread_idx=0;
    std::atomic<bool> is_using{false};
};

struct ggml_rknpu_matmul_part_B {
    rknn_matmul_shape shape;
    rknn_matmul_ctx ctx;
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;

    char name[GGML_NAME_MAX];
    int thread_idx;

    rknn_tensor_mem* B;
    bool B_is_copied = false;
};

struct ggml_rknpu_matmul_pair {
    ggml_rknpu_matmul_part_AC *part_AC;
    ggml_rknpu_matmul_part_B *part_B;

    ggml_rknpu_matmul_pair(ggml_rknpu_matmul_part_AC *part_AC, ggml_rknpu_matmul_part_B *part_B): part_AC(part_AC), part_B(part_B) {}

    ggml_rknpu_matmul_pair() {
        part_AC = NULL;
        part_B = NULL;
    }
};

static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

class rknn_timing_helper {
    // all time in microseconds
  public:
    double prepare_data_time_A   = 0.0;
    double prepare_data_time_B   = 0.0;
    double prepare_data_time_C   = 0.0;
    double memcpy_to_kernel_time = 0.0;
    double find_kernel_time      = 0.0;
    double set_io_time           = 0.0;
    double set_io_time_A         = 0.0;
    double set_io_time_B         = 0.0;
    double set_io_time_C         = 0.0;
    double run_time              = 0.0;
    double memcpy_to_result_time = 0.0;
    double free_pointer_time     = 0.0;
    double total_run_time        = 0.0;
    double quant_time            = 0.0;

    int rknn_threads = 1;

    rknn_timing_helper() {
        prepare_data_time_A   = 0.0;
        prepare_data_time_B   = 0.0;
        prepare_data_time_C   = 0.0;
        memcpy_to_kernel_time = 0.0;
        find_kernel_time      = 0.0;
        set_io_time           = 0.0;
        set_io_time_A         = 0.0;
        set_io_time_B         = 0.0;
        set_io_time_C         = 0.0;
        run_time              = 0.0;
        memcpy_to_result_time = 0.0;
        free_pointer_time     = 0.0;
        total_run_time        = 0.0;
        quant_time            = 0.0;

        timing_debug_printf("ggml-rknn: rknn_timing_helper initialized\n");
    }

    void clear() {
        prepare_data_time_A   = 0.0;
        prepare_data_time_B   = 0.0;
        prepare_data_time_C   = 0.0;
        memcpy_to_kernel_time = 0.0;
        find_kernel_time      = 0.0;
        set_io_time           = 0.0;
        set_io_time_A         = 0.0;
        set_io_time_B         = 0.0;
        set_io_time_C         = 0.0;
        run_time              = 0.0;
        memcpy_to_result_time = 0.0;
        free_pointer_time     = 0.0;
        total_run_time        = 0.0;
        quant_time            = 0.0;

        timing_debug_printf("ggml-rknn: rknn_timing_helper cleared\n");
    }

    void dump_time_usage() const {
        printf("\n");
        printf(" --- dump time usage avg for %d threads -----\n", this->rknn_threads);
        printf(" %8.0f us (%04.1f%%) prepare data time A B C: %.f, %.f, %.f us\n", \
            (this->prepare_data_time_A / this->rknn_threads + this->prepare_data_time_B + this->prepare_data_time_C / this->rknn_threads), \
            (this->prepare_data_time_A / this->rknn_threads + this->prepare_data_time_B + this->prepare_data_time_C / this->rknn_threads) / this->total_run_time * 100, \
            this->prepare_data_time_A / this->rknn_threads, \
            this->prepare_data_time_B, \
            this->prepare_data_time_C / this->rknn_threads);
        printf(" %8.0f us (%04.1f%%) quant time\n", (this->quant_time / this->rknn_threads), (this->quant_time / this->rknn_threads) / this->total_run_time * 100);
        printf(" %8.0f us (%04.1f%%) memcpy to kernel time\n", (this->memcpy_to_kernel_time / this->rknn_threads), (this->memcpy_to_kernel_time / this->rknn_threads) / this->total_run_time * 100);
        printf(" %8.0f us (%04.1f%%) find kernel time\n", (this->find_kernel_time / this->rknn_threads), (this->find_kernel_time / this->rknn_threads) / this->total_run_time * 100);
        printf(" %8.0f us (%04.1f%%) set io time\n", (this->set_io_time / this->rknn_threads), (this->set_io_time / this->rknn_threads) / this->total_run_time * 100);
        printf(" %8.0f us (%04.1f%%) set io time A B C: %.f, %.f, %.f us\n", \
            (this->set_io_time / this->rknn_threads), \
            (this->set_io_time / this->rknn_threads) / this->total_run_time * 100, \
            this->set_io_time_A / this->rknn_threads, \
            this->set_io_time_B / this->rknn_threads, \
            this->set_io_time_C / this->rknn_threads);
        printf(" %8.0f us (%04.1f%%) memcpy to result time\n", (this->memcpy_to_result_time / this->rknn_threads), (this->memcpy_to_result_time / this->rknn_threads) / this->total_run_time * 100);
        printf(" %8.0f us (%04.1f%%) run time\n", (this->run_time / this->rknn_threads), (this->run_time / this->rknn_threads) / this->total_run_time * 100);
        printf(" %8.0f us (%04.1f%%) free pointer time\n", (this->free_pointer_time / this->rknn_threads), (this->free_pointer_time / this->rknn_threads) / this->total_run_time * 100);
        printf(" %8.0f us (100.0%%) total_run_time\n", this->total_run_time);
    }
};

static rknn_timing_helper local_timer;

//MARK: QUANT HELPER

void custom_quantize_row_q8_0(const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT y, int64_t size, float &d) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    float amax = 0.0f; // absolute max

    // Find maximum absolute value with vectorization hints
    #pragma omp parallel for reduction(max:amax) num_threads(omp_threads) schedule(static)
    for (int j = 0; j < size; j++) {
        const float v = x[j];
        amax = MAX(amax, fabsf(v));
    }

    d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    // Quantize with vectorization hints and better memory access pattern
    #pragma omp parallel for num_threads(omp_threads) schedule(static)
    for (int j = 0; j < size; ++j) {
        const float x0 = x[j] * id;
        y[j] = (int8_t)roundf(x0);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // timing_debug_printf("ggml-rknn: custom_quantize_row_q8_0 duration: %ld us\n", duration.count());

    local_timer.quant_time += duration.count();
}

void custom_dequantize_row_q8_0(const int8_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int size, float d) {

    #pragma omp parallel for num_threads(omp_threads) schedule(static)
    for (int j = 0; j < size; ++j) {
        y[j] = x[j]*d;
    }
}

// MARK: Kernel def

#define GGML_RKNPU2_MAX_MATMUL_PARTS 512
static ggml_rknpu_matmul_part_AC matmul_parts_AC[GGML_RKNPU2_MAX_MATMUL_PARTS];
static ggml_rknpu_matmul_part_B matmul_parts_B[GGML_RKNPU2_MAX_MATMUL_PARTS];
static int matmul_parts_AC_count = 0;
static int matmul_parts_B_count = 0;

const char* rknpu2_matmul_type_to_string(rknn_matmul_type type)
{
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return "FLOAT16_MM_FLOAT16_TO_FLOAT32";
        case RKNN_INT8_MM_INT8_TO_INT32:
            return "INT8_MM_INT8_TO_INT32";
        case RKNN_INT4_MM_INT4_TO_INT16:
            return "INT4_MM_INT4_TO_INT16";
        default:
            GGML_ASSERT(0);
    }
}

size_t rknpu2_get_element_size(rknn_matmul_type type) {
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return sizeof(float16);
        case RKNN_INT8_MM_INT8_TO_INT32:
        case RKNN_INT4_MM_INT4_TO_INT16:
            return sizeof(int8_t);
        default:
            GGML_ASSERT(0);
    }
    return 0;
}

static const char *get_dims_string(rknn_matmul_tensor_attr *attr)
{
  if (!attr->n_dims)
  {
    return "()";
  }
  static char dims_str[128];
  memset(&dims_str[0], 0, sizeof(dims_str));
  sprintf(&dims_str[0], "(%d", attr->dims[0]);
  for (uint32_t i = 1; i < attr->n_dims; ++i)
  {
    int idx = strlen(dims_str);
    sprintf(&dims_str[idx], ", %d", attr->dims[i]);
  }
  strcat(&dims_str[0], ")");
  return dims_str;
}

static void dump_matmul_tensor_attr(rknn_matmul_tensor_attr *attr)
{
  timing_debug_printf("  name=%s, dims=%s, size=%d, type=%s\n", attr->name, get_dims_string(attr), attr->size,
         get_type_string(attr->type));
}

struct ggml_rknpu_matmul_part_AC * ggml_rknpu_matmul_part_AC_find(int m, int k, int n, rknn_matmul_type type, int thread_idx, bool is_prefill=false){
    timing_debug_printf("ggml-rknn: ggml_rknpu_matmul_part_AC_find: m: %d, k: %d, n: %d, type: %s, thread_idx: %d\n", m, k, n, rknpu2_matmul_type_to_string(type), thread_idx);
    for (int i = 0; i < matmul_parts_AC_count; i++) {
        ggml_rknpu_matmul_part_AC *part = &matmul_parts_AC[i];
        if (part->M == m && part->K == k && part->N == n && part->type == type && part->thread_idx == thread_idx && part->is_using == false && part->prefill == is_prefill) {
            return part;
        }
    }
    return NULL;
}

struct ggml_rknpu_matmul_part_B * ggml_rknpu_matmul_part_B_find(const char * name, int thread_idx){
    if (name == NULL) {
        timing_debug_printf("ggml-rknn: ggml_rknpu_matmul_part_B_find: name is NULL\n");
        return NULL;
    }

    timing_debug_printf("ggml-rknn: ggml_rknpu_matmul_part_B_find: name: %s, thread_idx: %d\n", name, thread_idx);

    for (int i = 0; i < matmul_parts_B_count; i++) {
        ggml_rknpu_matmul_part_B *part = &matmul_parts_B[i];
        if (strncmp(part->name, name, GGML_NAME_MAX) == 0) {
            if(part->thread_idx == thread_idx) {
                return part;
            }
        }
    }
    return NULL;
}

struct ggml_rknpu_matmul_pair create_matmul_pair(int M, int K, int N, rknn_matmul_type type, int thread_idx, char * name, int core_mask_bits, int ac_layout_perf=1) {
    // core_mask_bits: bitmask of allowed NPU cores (bit0=CORE_0, bit1=CORE_1, bit2=CORE_2)
    // ac_layout_perf bitmask: bit0=decode_perf, bit1=prefill_perf
    //   0=off for both, 1=decode on only, 2=prefill on only, 3=both on
    const bool decode_perf  = (ac_layout_perf & 1) != 0;
    const bool prefill_perf = (ac_layout_perf & 2) != 0;
    const bool is_prefill_current = (M > DECODE_MAX_M);

    if (name == NULL) {
        timing_debug_printf("ggml-rknn: create_matmul_pair: name is NULL\n");
        name = strdup("unknown");
    }

    timing_debug_printf("ggml-rknn: create_matmul_pair: name: %s, thread_idx: %d, core_mask_bits: 0x%x decode_perf=%d prefill_perf=%d is_prefill=%d\n",
                        name, thread_idx, core_mask_bits, decode_perf, prefill_perf, is_prefill_current);

    ggml_rknpu_matmul_part_AC *part_AC = ggml_rknpu_matmul_part_AC_find(M, K, N, type, thread_idx, is_prefill_current);
    ggml_rknpu_matmul_part_B *part_B = ggml_rknpu_matmul_part_B_find(name, thread_idx);

    // Assign thread to the thread_idx-th set bit in core_mask_bits.
    // e.g. core_mask_bits=0x6 (CORE_1+CORE_2): thread 0->CORE_1, thread 1->CORE_2
    rknn_core_mask core_mask = (rknn_core_mask)npu_core_for_thread(core_mask_bits, thread_idx);
    npu_core_debug_printf("[NPU_CORE] create_matmul_pair: name=%s thread_idx=%d core_mask_bits=0x%x -> core_mask=0x%x (%s)\n",
                          name, thread_idx, core_mask_bits, (int)core_mask, core_mask_to_str((int)core_mask));

    timing_debug_printf("ggml-rknn: creating matmul pair for %s:%d size %d x %d x %d\n", name, thread_idx, M, K, N);

    if (part_B == NULL) {
        timing_debug_printf("ggml-rknn: no B found, creating new B for %s:%d size %d x %d x %d\n", name, thread_idx, M, K, N);
        if (matmul_parts_B_count >= GGML_RKNPU2_MAX_MATMUL_PARTS) {
            fprintf(stderr, "ggml-rknn: matmul_parts_B_count too much part_B \n");
            GGML_ASSERT(0);
        }

        // Add a mutex at file/class level
        static std::mutex matmul_parts_B_mutex;

        // Then protect the increment:
        std::lock_guard<std::mutex> lock(matmul_parts_B_mutex);

        part_B = &matmul_parts_B[matmul_parts_B_count++];
        memset(part_B, 0, sizeof(ggml_rknpu_matmul_part_B));
        
        strncpy(part_B->name, name, GGML_NAME_MAX);
        part_B->thread_idx = thread_idx;
        part_B->B_is_copied = false;

        // if prefill, create another io_attr in AC 

        memset(&part_B->info, 0, sizeof(rknn_matmul_info));
        part_B ->info.M = M;
        part_B ->info.K = K;
        part_B ->info.N = N;
        part_B ->info.type = type;
        part_B ->info.B_layout = 1; // B use native layout (weight)
        part_B ->info.AC_layout = decode_perf ? 1 : 0; // A and C layout for decode phase

        memset(&part_B->io_attr, 0, sizeof(rknn_matmul_io_attr));

        int ret = rknn_matmul_create(&(part_B->ctx), &(part_B->info), &(part_B->io_attr));
        GGML_ASSERT(ret == 0);

        rknn_matmul_set_core_mask(part_B->ctx, core_mask);
        npu_core_debug_printf("[NPU_CORE] rknn_matmul_set_core_mask (new B): name=%s thread=%d mask=0x%x (%s)\n",
                              name, thread_idx, (int)core_mask, core_mask_to_str((int)core_mask));

        #if GGML_RKNPU2_USE_OUTSIDE_ALLOC
            int fd = -1;
            uint8_t *va = NULL;
            dma_alloc(part_B->io_attr.B.size, &fd, (void **)&va);
            dma_sync_device_to_cpu(fd);
            part_B->B = rknn_create_mem_from_fd(part_B->ctx, fd, va,
                                            part_B->io_attr.B.size, 0);

        #else
            part_B->B =
                rknn_create_mem(part_B->ctx, part_B->io_attr.B.size);
        #endif
        
        // // Protect rknpu2_allocated_bytes with mutex
        // static std::mutex rknpu2_allocated_bytes_mutex;
        // {
        //     std::lock_guard<std::mutex> alloc_lock_b(rknpu2_allocated_bytes_mutex);
        //     rknpu2_allocated_bytes += part_B->io_attr.B.size;
        // }
        if (part_B->B == NULL) {
            fprintf(stderr, "ggml-rknn: rknn_create_mem failed for B node %s:%d size %u\n", name, thread_idx, part_B->io_attr.B.size);
            fprintf(stderr, "ggml-rknn: allocated bytes: %lu, max memory: %llu\n", rknpu2_allocated_bytes, MAX_RKNN_MEMORY);
            GGML_ASSERT(0);
        }

        timing_debug_printf("ggml-rknn: created B node %s:%d; bytes: %ud\n", name, thread_idx, part_B->io_attr.B.size);
    } else {
        rknn_matmul_set_core_mask(part_B->ctx, core_mask);
        npu_core_debug_printf("[NPU_CORE] rknn_matmul_set_core_mask (existing B): name=%s thread=%d mask=0x%x (%s)\n",
                              name, thread_idx, (int)core_mask, core_mask_to_str((int)core_mask));

        timing_debug_printf("ggml-rknn: found B node %s:%d size %d x %d x %d\n", part_B->name, part_B->thread_idx, part_B->info.M, part_B->info.K, part_B->info.N);
    }
    GGML_ASSERT(part_B != NULL);

    if (part_AC == NULL) {
        timing_debug_printf("ggml-rknn: no AC found, creating new AC for %s:%d size %d x %d x %d\n", name, thread_idx, M, K, N);

        static std::mutex matmul_parts_AC_mutex;
        static std::mutex rknpu2_allocated_bytes_mutex;
        std::lock_guard<std::mutex> lock(matmul_parts_AC_mutex);

        // Helper: allocate and initialize one part_AC entry.
        // is_prefill_variant=true  → own rknn_matmul ctx with use_perf AC_layout (prefill path).
        // is_prefill_variant=false → shared part_B ctx whose AC_layout was baked in at part_B creation
        //                            (decode_perf); only valid when M == part_B->info.M.
        auto create_one_ac = [&](bool is_prefill_variant, bool use_perf) -> ggml_rknpu_matmul_part_AC* {
            if (matmul_parts_AC_count >= GGML_RKNPU2_MAX_MATMUL_PARTS) {
                fprintf(stderr, "ggml-rknn: matmul_parts_AC_count too much part_AC \n");
                GGML_ASSERT(0);
            }
            ggml_rknpu_matmul_part_AC *ac = &matmul_parts_AC[matmul_parts_AC_count++];
            memset(ac, 0, sizeof(ggml_rknpu_matmul_part_AC));
            ac->M = M;
            ac->K = K;
            ac->N = N;
            ac->type = type;
            ac->thread_idx = thread_idx;
            ac->ac_layout_perf = use_perf;
            ac->prefill = is_prefill_variant;

            if (is_prefill_variant) {
                timing_debug_printf("ggml-rknn: creating prefill-variant AC for %s:%d size %d x %d x %d ac_perf=%d\n",
                                    name, thread_idx, M, K, N, use_perf);
                rknn_matmul_info info;
                memset(&info, 0, sizeof(rknn_matmul_info));
                info.M = M; info.K = K; info.N = N; info.type = type;
                info.B_layout = 1;
                info.AC_layout = use_perf ? 1 : 0;
                rknn_matmul_io_attr io_attr;
                memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));
                int ret = rknn_matmul_create(&ac->ctx, &info, &io_attr);
                if (ret < 0) {
                    fprintf(stderr, "ggml-rknn: rknn_matmul_create failed for prefill-variant AC %s:%d %d x %d\n",
                            name, thread_idx, M, K);
                    GGML_ASSERT(0);
                }
                rknn_set_core_mask(ac->ctx, core_mask);
                npu_core_debug_printf("[NPU_CORE] rknn_set_core_mask (new AC prefill-variant): name=%s thread=%d mask=0x%x (%s)\n",
                                      name, thread_idx, (int)core_mask, core_mask_to_str((int)core_mask));
                ac->io_attr = io_attr;
            } else {
                timing_debug_printf("ggml-rknn: creating decode-variant AC for %s:%d size %d x %d x %d ac_perf=%d\n",
                                    name, thread_idx, M, K, N, use_perf);
                ac->ctx = part_B->ctx;
                ac->io_attr = part_B->io_attr;
            }

            ac->A = rknn_create_mem(ac->ctx, ac->io_attr.A.size);
            {
                std::lock_guard<std::mutex> alloc_lock(rknpu2_allocated_bytes_mutex);
                rknpu2_allocated_bytes += ac->io_attr.A.size;
            }
            if (ac->A == NULL) {
                fprintf(stderr, "ggml-rknn: rknn_create_mem failed for A %s:%d bytes: %u %d x %d\n",
                        name, thread_idx, ac->io_attr.A.size, M, K);
                GGML_ASSERT(0);
            }
            ac->C = rknn_create_mem(ac->ctx, ac->io_attr.C.size);
            {
                std::lock_guard<std::mutex> alloc_lock(rknpu2_allocated_bytes_mutex);
                rknpu2_allocated_bytes += ac->io_attr.C.size;
            }
            if (ac->C == NULL) {
                fprintf(stderr, "ggml-rknn: rknn_create_mem failed for C %s:%d bytes: %u %d x %d\n",
                        name, thread_idx, ac->io_attr.C.size, M, N);
                GGML_ASSERT(0);
            }
            timing_debug_printf("ggml-rknn: created AC (prefill=%d ac_perf=%d) for %s:%d bytes: %ud\n",
                                 is_prefill_variant, use_perf, name, thread_idx,
                                 ac->io_attr.A.size + ac->io_attr.C.size);
            return ac;
        };

        // Create the primary part_AC for the current call's phase.
        bool current_perf = is_prefill_current ? prefill_perf : decode_perf;
        part_AC = create_one_ac(is_prefill_current, current_perf);

        // If decode and prefill have different settings, pre-create the alternate variant
        // so it is ready when the other phase first runs.
        // Decode-variant alternate is only valid when M == part_B->info.M (shared ctx).
        if (decode_perf != prefill_perf) {
            bool alt_is_prefill = !is_prefill_current;
            bool alt_perf       = alt_is_prefill ? prefill_perf : decode_perf;
            bool can_create_alt = alt_is_prefill || (M == part_B->info.M);
            if (can_create_alt) {
                ggml_rknpu_matmul_part_AC *alt = ggml_rknpu_matmul_part_AC_find(M, K, N, type, thread_idx, alt_is_prefill);
                if (alt == NULL) {
                    timing_debug_printf("ggml-rknn: pre-creating alternate AC (prefill=%d ac_perf=%d) for %s:%d\n",
                                        alt_is_prefill, alt_perf, name, thread_idx);
                    create_one_ac(alt_is_prefill, alt_perf);
                }
            }
        }
    } else {
        rknn_set_core_mask(part_AC->ctx, core_mask);
        npu_core_debug_printf("[NPU_CORE] rknn_set_core_mask (existing AC): name=%s thread=%d mask=0x%x (%s)\n",
                              name, thread_idx, (int)core_mask, core_mask_to_str((int)core_mask));

        // found AC
        timing_debug_printf("ggml-rknn: found AC for %s:%d size %d x %d x %d\n", name, thread_idx, M, K, N);
        if (part_AC->prefill) {
            //TODO: 
            // printf("ggml-rknn: found prefill kernel case\n");
            // part_AC->ctx = part_B->ctx;
            // GGML_ASSERT(part_AC->M == part_B->info.M);

            // GGML_ASSERT(part_AC->K == part_B->info.K);
            // GGML_ASSERT(part_AC->N == part_B->info.N);
        } else {
            part_AC->ctx = part_B->ctx;
        }

        if (part_AC->K != part_B->info.K || part_AC->N != part_B->info.N || part_AC->thread_idx != part_B->thread_idx) {
            printf("ggml-rknn: prefill kernel %s:%d case K or N or thread_idx mismatch: %d != %d or %d != %d or %d != %d\n", name, thread_idx, part_AC->K, part_B->info.K, part_AC->N, part_B->info.N, part_AC->thread_idx, part_B->thread_idx);
            GGML_ASSERT(0);
        }
    }
    GGML_ASSERT(part_AC != NULL);

    timing_debug_printf("ggml-rknn: created matmul pair for %s:%d size %d x %d x %d\n", name, thread_idx, M, K, N);
    timing_debug_printf("ggml-rknn: part_AC->ctx: %p, part_B->ctx: %p\n", part_AC->ctx, part_B->ctx);
    
    return ggml_rknpu_matmul_pair(part_AC, part_B);
}

// Given an integer pointer (array) and a length, concatenate to string as "a,b,c"
static std::string dim_array_to_string(const uint32_t *arr, int len) {
    std::string result;
    for (int i = 0; i < len; ++i) {
        result += std::to_string(arr[i]);
        if (i != len - 1) {
            result += ",";
        }
    }
    return result;
}




//MARK: debug helper
template <typename T>
void printMatrix(const T * matrix, int rows, int cols, const char * name = "Matrix", bool is_float = true) {
    printf("%s (%d x %d): \n[", name, rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // Adjust format specifier based on type T if needed, for now assume float/int compatible
            //   if constexpr (std::is_floating_point_v<T>) {
            if (is_float) {
                printf("%8.4f ", (float) (((T *) matrix)[r * cols + c]));
            } else {
                printf("%8d ", (T) matrix[r * cols + c]);
            }
        }
        printf("\n");
    }
    printf("]\n");
}


//MARK: DEMO HELPER
// static const char *get_dims_string(rknn_matmul_tensor_attr *attr) {
//     if (!attr->n_dims) {
//         return "()";
//     }
//     static char dims_str[128];
//     memset(&dims_str[0], 0, sizeof(dims_str));
//     sprintf(&dims_str[0], "(%d", attr->dims[0]);
//     for (uint32_t i = 1; i < attr->n_dims; ++i) {
//         int idx = strlen(dims_str);
//         sprintf(&dims_str[idx], ", %d", attr->dims[i]);
//     }
//     strcat(&dims_str[0], ")");
//     return dims_str;
// }

static int8_t get_virt_addr_int4(void *virt_addr, int index) {
    int8_t int4 = 0;
    if (index % 2 == 0) {
        int4 = (((int8_t *)virt_addr)[index / 2] >> 4) & 0xf;
    } else {
        int4 = (((int8_t *)virt_addr)[index / 2]) & 0xf;
    }
    if (int4 & 0x8) {
        int4 = int4 | 0xf0;
    }
    return int4;
}

static void dump_matmul_tensor(rknn_tensor_mem * tensor, rknn_matmul_tensor_attr * attr) {
    printf("  %s%s:\n", attr->name, get_dims_string(attr));
    // normal layout
    if (attr->n_dims == 2) {
        for (uint32_t i = 0; i < attr->dims[0]; ++i) {
            for (uint32_t j = 0; j < attr->dims[1]; ++j) {
                void * virt_addr = (void *) ((size_t) tensor->virt_addr + tensor->offset);
                if (attr->type == RKNN_TENSOR_INT8) {
                    printf(" %4d", ((int8_t *) virt_addr)[i * attr->dims[1] + j]);
                } else if (attr->type == RKNN_TENSOR_INT32) {
                    printf(" %6d", ((int32_t *) virt_addr)[i * attr->dims[1] + j]);
                } else if (attr->type == RKNN_TENSOR_FLOAT16) {
                    printf(" %5.2f", (float) (((float16 *) virt_addr)[i * attr->dims[1] + j]));
                } else if (attr->type == RKNN_TENSOR_FLOAT32) {
                    printf(" %5.2f", ((float *) virt_addr)[i * attr->dims[1] + j]);
                } else if (attr->type == RKNN_TENSOR_INT16) {
                    printf(" %d", ((int16_t *) virt_addr)[i * attr->dims[1] + j]);
                } else if (attr->type == RKNN_TENSOR_INT4) {
                    int    index = i * attr->dims[1] + j;
                    int8_t int4  = get_virt_addr_int4(virt_addr, index);
                    printf("%d ", int4);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
    // perf layout
    else if (attr->n_dims == 3) {
        for (uint32_t i = 0; i < attr->dims[0]; ++i) {
            for (uint32_t j = 0; j < attr->dims[1]; ++j) {
                for (uint32_t k = 0; k < attr->dims[2]; ++k) {
                    void * virt_addr = (void *) ((size_t) tensor->virt_addr + tensor->offset);
                    if (attr->type == RKNN_TENSOR_INT4) {
                    int    index = (i * attr->dims[1] + j) * attr->dims[2] + k;
                    int8_t int4  = get_virt_addr_int4(virt_addr, index);
                    printf("%d ", int4);
                    } else if (attr->type == RKNN_TENSOR_INT8) {
                    printf(" %4d ", ((int8_t *) virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
                    } else if (attr->type == RKNN_TENSOR_INT16) {
                    printf(" %6d ", ((int16_t *) virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
                    } else if (attr->type == RKNN_TENSOR_INT32) {
                    printf(" %6d ", ((int32_t *) virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
                    } else if (attr->type == RKNN_TENSOR_FLOAT16) {
                    printf(" %5.2f ", (float) (((float16 *) virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]));
                    } else if (attr->type == RKNN_TENSOR_FLOAT32) {
                    printf(" %5.2f ", ((float *) virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
                    }
                }
                printf("\n");
            }
            printf("\n");
        }
    }
    // native layout
    else if (attr->n_dims == 4) {
        // N / 16
        for (uint32_t n = 0; n < attr->dims[0]; ++n) {
            // K / 32
            for (uint32_t k = 0; k < attr->dims[1]; ++k) {
                // 16
                for (uint32_t nn = 0; nn < attr->dims[2]; ++nn) {
                    // 32
                    for (uint32_t kk = 0; kk < attr->dims[3]; kk++) {
                    void * virt_addr = (void *) ((size_t) tensor->virt_addr + tensor->offset);
                    if (attr->type == RKNN_TENSOR_INT4) {
                        int    index = ((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk;
                        int8_t int4  = get_virt_addr_int4(virt_addr, index);
                        printf("%d ", int4);
                    } else if (attr->type == RKNN_TENSOR_INT8) {
                        printf(" %4d ",
                               ((int8_t *)
                                    virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]);
                    } else if (attr->type == RKNN_TENSOR_INT32) {
                        printf(" %6d ",
                               ((int32_t *)
                                    virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]);
                    } else if (attr->type == RKNN_TENSOR_FLOAT16) {
                        printf(
                            " %5.2f ",
                            (float) ((
                                (float16 *)
                                    virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]));
                    } else if (attr->type == RKNN_TENSOR_FLOAT32) {
                        printf(
                            " %5.2f ",
                            ((float *) virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]);
                    }
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

//MARK: LAYOUT HELPER
/**
    * @brief convert norm layout to perf layout
    * column major -> row major
    * norm layout (GGML): [K,M] (calling [k + m * K])
    * norm layout (RKNN): [M,K] (calling [m * K + k])
    * perf layout (RKNN): [K/subK, M, subK] (calling [(ksk * M + m) * subK + j])
    * for RK3588 we align to 16, FP16 subK=8, INT8 subK=16 

    * for calibrating the indices: 
    * normal layout (human readable): A[m][k] 
    * perf layout (RKNN): it's located in A_perf[floor(k/subK)][m][k % subK]
    */
template <typename Ti, typename To>
void norm_layout_to_perf_layout_A(Ti * src, To * dst, int32_t M, int32_t K, int32_t subK, bool isInt4Type = false) {
    int outter_size = (int) std::ceil(K * 1.0f / subK);
    for (int i = 0; i < outter_size; i++) {
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < subK; j++) {
                int ki = i * subK + j;
                if (isInt4Type) {
                    int    input_index  = m * K + ki;
                    int    output_index = i * M * subK + m * subK + j;
                    int8_t int4         = src[input_index];
                    if (ki >= K) {
                        int4 = 0;
                    } else {
                        int4 = int4 & 0xf;
                    }
                    if (output_index % 2 == 0) {
                        dst[output_index / 2] = int4 << 4;
                    } else {
                        dst[output_index / 2] = (int4 | (int8_t) (dst[output_index / 2]));
                    }
                } else {
                    // if (ki >= K) {
                    //     dst[i * M * subK + m * subK + j] = 0;
                    // } else {
                        
                    dst[i * M * subK + m * subK + j] = src[m * K + ki];
                    // }
                }
            }
        }
    }
}

void ggml_layout_to_perf_layout_A(const float * src, rknpu2::float16 * dst, int32_t M, int32_t K, int32_t subK) {
    
    int outer_size = (K + subK - 1) / subK;
    
    // Parallelize the outer loop over k1 (K dimension blocks)
    #pragma omp parallel for num_threads(omp_threads) schedule(static)
    for (int k1 = 0; k1 < outer_size; k1++) {
        int base_offset = k1 * M * subK;
        int k_base = k1 * subK;
        
        for (int m = 0; m < M; m++) {
            // Source: contiguous subK floats starting at src[m*K + k_base]
            const float * src_ptr = src + (size_t)m * K + k_base;
            // Dest: contiguous subK float16s at dst[base_offset + m*subK]
            int dst_offset = base_offset + m * subK;
            uint16_t * dst_u16 = (uint16_t *)(dst + dst_offset);
            
            int j = 0;
#ifdef __ARM_NEON
            // Convert 8 floats at a time using NEON F32->F16 hardware conversion
            for (; j + 7 < subK; j += 8) {
                float32x4_t f32_lo = vld1q_f32(src_ptr + j);
                float32x4_t f32_hi = vld1q_f32(src_ptr + j + 4);
                float16x8_t f16_vec = vcombine_f16(vcvt_f16_f32(f32_lo), vcvt_f16_f32(f32_hi));
                vst1q_u16(dst_u16 + j, vreinterpretq_u16_f16(f16_vec));
            }
            // Convert remaining 4 floats
            for (; j + 3 < subK; j += 4) {
                float32x4_t f32_vec = vld1q_f32(src_ptr + j);
                float16x4_t f16_vec = vcvt_f16_f32(f32_vec);
                vst1_u16(dst_u16 + j, vreinterpret_u16_f16(f16_vec));
            }
#endif
            // Scalar remainder
            for (; j < subK; j++) {
                dst[dst_offset + j] = (rknpu2::float16)(src_ptr[j]);
            }
        }
    }
}

/**
     * @brief convert norm layout to native layout
     * norm layout:  [K,N]
     * native layout: [N1, K1, subN, subK]
     * for RK3588, FP16 subK=32 subN=16, INT8 subK=32 subN=32
     *
     */
template <typename Ti, typename To>
void norm_layout_to_native_layout_B(Ti * src, To * dst, int32_t K, int32_t N, int32_t subN, int32_t subK,
                                  bool isInt4Type) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int N_remain = (int) std::ceil(N * 1.0f / subN);
    int K_remain = (int) std::ceil(K * 1.0f / subK);
    for (int i = 0; i < N_remain; i++) {
        for (int j = 0; j < K_remain; j++) {
            for (int n = 0; n < subN; n++) {
                int ni = i * subN + n;
                for (int k = 0; k < subK; k++) {
                    int ki = j * subK + k;
                    if (isInt4Type) {
                        int    input_index  = ki * N + ni;
                        int    output_index = i * (K_remain * subN * subK) + j * (subN * subK) + n * subK + k;
                        int8_t int4         = src[input_index];
                        if (ki < K && ni < N) {
                            int4 = int4 & 0xf;
                        } else {
                            int4 = 0;
                        }
                        if (output_index % 2 == 0) {
                            dst[output_index / 2] = int4 << 4;
                        } else {
                            int8_t temp           = dst[output_index / 2];
                            int8_t result         = temp | int4;
                            dst[output_index / 2] = result;
                        }
                    } else {
                        if (ki < K && ni < N) {
                            dst[((i * K_remain + j) * subN + n) * subK + k] = src[ki * N + ni];
                        } else {
                            dst[((i * K_remain + j) * subN + n) * subK + k] = 0;
                        }
                    }
                }
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    timing_debug_printf("ggml-rknn: norm_layout_to_native_layout duration: %ld us\n", duration);
}

template void norm_layout_to_native_layout_B<int8_t, int8_t>(int8_t * src, int8_t * dst, int32_t K, int32_t N,
                                                           int32_t subN, int32_t subK, bool isInt4Type);
template void norm_layout_to_native_layout_B<float16, float16>(float16 * src, float16 * dst, int32_t K, int32_t N,
                                                             int32_t subN, int32_t subK, bool isInt4Type);

/**
     * @brief convert FP16 ggml layout to native layout
     * ggml layout:  [N, K]
     * native layout: [N1, K1, subN, subK]
     *
     * note here ggml layout is column major, 

        -  normal layout: (K, N) (row major)
              [K1N1, K1N2, ..., K1Nn, --- continuous this direction ---
               K2N1, K2N2, ..., K2Nn,
               ...
               KkN1, KkN2, ..., KkNn]

              float16: for RK3588

        -   native layout: (N / 16, K / 32, 16, 32) (row major)
              [K1N1,  K2N1,  ..., K32N1, --- continuous this direction ---
               K1N2,  K2N2,  ..., K32N2,
               ...
               K1N16, K2N16, ..., K32N16,
               K33N1, K34N1, ..., K64N1,
               K33N2, K34N2, ..., K64N2,
               ...
               K(k-31)N16, K(k-30)N16, ..., KkN16,
               K1N17, K2N17, ..., K32N17,
               K1N18, K2N18, ..., K32N18,
               ...
               K(k-31)Nn, K(k-30)Nn, ..., KkNn]
        -   ggml layout: (N, K) (column major)
              [N1K1, N1K2, ..., N1Kn, 
               N2K1, N2K2, ..., N2Kn,
               ...
               NkK1, NkK2, ..., NkKn]
               |
               |
               continuous this direction 
     */
template <typename Ti, typename To>
void ggml_layout_to_native_layout_B(Ti * src, To * dst, int32_t K, int32_t N, int32_t subN, int32_t subK,
                                  bool isInt4Type) {
    auto start_time = std::chrono::high_resolution_clock::now();
    int  N_remain   = (int) std::ceil(N * 1.0f / subN);
    int  K_remain   = (int) std::ceil(K * 1.0f / subK);
    for (int i = 0; i < N_remain; i++) {
        for (int j = 0; j < K_remain; j++) {
            for (int n = 0; n < subN; n++) {
                int ni = i * subN + n;
                for (int k = 0; k < subK; k++) {
                    int ki = j * subK + k;
                    if (isInt4Type) {
                        int    input_index  = ki * N + ni;
                        int    output_index = i * (K_remain * subN * subK) + j * (subN * subK) + n * subK + k;
                        int8_t int4         = src[input_index];
                        if (ki < K && ni < N) {
                            int4 = int4 & 0xf;
                        } else {
                            int4 = 0;
                        }
                        if (output_index % 2 == 0) {
                            dst[output_index / 2] = int4 << 4;
                        } else {
                            int8_t temp           = dst[output_index / 2];
                            int8_t result         = temp | int4;
                            dst[output_index / 2] = result;
                        }
                    } else {
                        //TODO: parallelize it 
                        memcpy(dst + ((i * K_remain + j) * subN + n) * subK, src + j * subK + K * ni, sizeof(To) * subK );

                        // if (ki < K && ni < N) {
                        //     dst[((i * K_remain + j) * subN + n) * subK + k] = src[ki + K * ni];
                        // } else {
                        //     dst[((i * K_remain + j) * subN + n) * subK + k] = 0;
                        // }
                    }
                }
            }
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    timing_debug_printf("ggml-rknn: ggml_layout_to_native_layout duration: %ld us\n", duration);
}

template <typename Ti, typename To>
void ggml_layout_to_native_layout_B_multithreads(Ti * src, To * dst, int32_t K, int32_t N, int32_t subN, int32_t subK, int num_threads = omp_threads) {
    // only supports fp16
    // int num_threads = std::thread::hardware_concurrency();
    // int num_threads = 4;

    // auto start_time = std::chrono::high_resolution_clock::now();


    int N_remain = (N + subN - 1) / subN;
    int K_remain = (K + subK - 1) / subK;

    std::vector<std::thread> threads;
    
    auto worker = [&](int thread_id) {
        int blocks_per_thread = (N_remain + num_threads - 1) / num_threads;
        int start_i = thread_id * blocks_per_thread;
        int end_i = std::min((thread_id + 1) * blocks_per_thread, N_remain);
        
        for (int i = start_i; i < end_i; i++) {
            for (int j = 0; j < K_remain; j++) {
                for (int n = 0; n < subN; n++) {
                    int ni = i * subN + n;
                    for (int k = 0; k < subK; k++) {
                        memcpy(dst + ((i * K_remain + j) * subN + n) * subK, src + j * subK + K * ni, sizeof(To) * subK );
                    }
                }
            }
        }
    };
    
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // timing_debug_printf("ggml-rknn: ggml_layout_to_native_layout_multithreads duration: %d threads %ld us\n", num_threads, duration);
}

/**
     * @brief convert Q8_0 ggml layout to native layout
     * ggml layout:  [N, K]
     * native layout: [N1, K1, subN, subK]
     *
     * note here ggml layout is column major, 

        -  normal layout: (K, N) (row major) (human normal)
            [K0N0, K0N1, ..., K0N(n-1), --- continuous this direction ---
            K1N0, K1N1, ..., K1N(n-1),
            ...
            K(k-1)N0, K(k-1)N1, ..., K(k-1)N(n-1)]

        -   Q8_0: 32 

        -   native layout: (N / 32, K / 32, 32, 32) (row major)
            [K0N0,  K1N0,  ..., K31N0, --- continuous this direction ---
            K0N1,  K1N1,  ..., K31N1,
            ...
            K0N31, K1N31, ..., K31N31,
            K32N0, K33N0, ..., K63N0,
            K32N1, K33N1, ..., K63N1,
            ...
            K(k-32)N31, K(k-31)N31, ..., K(k-1)N31,
            K0N16, K1N16, ..., K31N16,
            K0N17, K1N17, ..., K31N17,
            ...
            K(k-32)N(n-1), K(k-31)N(n-1), ..., K(k-1)N(n-1)]

        -   ggml layout: (K * N / 32, 32)
            [
                block[0]: [q0, q1, q2, ..., q31], d 
                        (K0N0, K1N0, ..., K31N0)
                block[1]: [q0, q1, q2, ..., q31], d 
                        (K32N0, K33N0, ..., K63N0)
                ...
                block[K/32]: [q0, q1, q2, ..., q31], d
                        (K0N1, K1N1, ..., K31N1)
                ...
                block[num_blocks-1]: [q0, q1, q2, ..., q31], d
                        (K(K-31)N(N-1), K(K-30)N(N-1), ..., K(K-1)N(N-1))
            ]

            KkNn is located in 
                Block: block[ n * (K / 32) + floor(k / 32) ]
                Quantization Value: qs[ k % 32 ]

                

        - for calibrating and finding elements, 
        to find B[k][n],
        it's in ggml q8's Block[n * (K / 32) + floor(k / 32)].qs[k % 32]

        in rknn B_native[floor(n/subN)][floor(k/subK)][n % subN][k % subK]
        since I quantize by dimention K, B[k][0...N-1]'s delta is delta[k]

        // Copy-pasted from ggml.c

        #define QK8_0 32
        typedef struct {
            rknpu2::float16 d;   // delta
            int8_t  qs[QK8_0];   // quants
        } block_q8_0;
     */

void ggml_layout_to_native_layout_B_q8_0(block_q8_0 * src, int8_t * dst, rknpu2::float16 * delta, int32_t K, int32_t N, int32_t subN = 32, int32_t subK = 32) {
    // only supports q8
    // QK8_0 = 32;

    // where to save delta? (scaling factor)

    // int num_blocks = (K * N) / QK8_0;
    GGML_ASSERT(subK == QK8_0);

    // integer ceil
    int  N_remain   = (N + subN - 1) / subN;
    int  K_remain   = (K + subK - 1) / subK;
    for (int i = 0; i < N_remain; i++) {
        for (int j = 0; j < K_remain; j++) {
            for (int n = 0; n < subN; n++) {
                // int ni = i * subN + n;
                // magic below: 
                int block_index = (i * subN + n) * (K / subK) + j;
                //TODO: parallelize it 
                memcpy(dst + ((i * K_remain + j) * subN + n) * subK, src[block_index].qs, sizeof(int8_t) * QK8_0 );

                // delta is the scaling factor, it keeps the order of blocks instead of native layout order
                delta[block_index] = src[block_index].d;
            }
        }
    }
}

void ggml_layout_to_native_layout_B_custom_q8_0(block_q8_0 * src, int8_t * dst, float * delta, int32_t K, int32_t N, int32_t subN = 32, int32_t subK = 32) {
    // only supports q8
    // QK8_0 = 32;
    
    float* temp = new float[K];
    int8_t* temp_qs = new int8_t[K];

    // int N_remain = (N + subN - 1) / subN;
    int K_remain = (K + subK - 1) / subK;

    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k += QK8_0) {
            custom_dequantize_row_q8_0(src[(n * K + k) / QK8_0].qs, temp + k, QK8_0, ggml_compute_fp16_to_fp32(src[(n * K + k) / QK8_0].d));
        }
        custom_quantize_row_q8_0(temp, temp_qs, K, delta[n]);

        for (int k = 0; k < K; k += subK) {
            memcpy(dst + (((n / subN) * K_remain + (k / subK)) * subN + (n % subN)) * subK, temp_qs + k, subK * sizeof(int8_t));
        }
    }
    delete [] temp;
    delete [] temp_qs;
    
}

void ggml_layout_to_native_layout_B_custom_q8_0_multithread(block_q8_0 * src, int8_t * dst, float * delta, int32_t K, int32_t N, int32_t subN = 32, int32_t subK = 32) {
    // only supports q8
    // QK8_0 = 32;
    
    const int num_threads = std::min(N, (int)omp_threads);
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    
    // Pre-allocate per-thread buffers to avoid contention
    std::vector<std::unique_ptr<float[]>> thread_temps(num_threads);
    std::vector<std::unique_ptr<int8_t[]>> thread_temp_qs(num_threads);
    
    for (int t = 0; t < num_threads; t++) {
        thread_temps[t] = std::make_unique<float[]>(K);
        thread_temp_qs[t] = std::make_unique<int8_t[]>(K);
    }
    
    // int N_remain = (N + subN - 1) / subN;
    int K_remain = (K + subK - 1) / subK;
    
    auto worker = [&](int thread_id) {
        float* temp = thread_temps[thread_id].get();
        int8_t* temp_qs = thread_temp_qs[thread_id].get();
        
        // Distribute work across threads
        const int n_per_thread = (N + num_threads - 1) / num_threads;
        const int start_n = thread_id * n_per_thread;
        const int end_n = std::min(start_n + n_per_thread, N);
        
        for (int n = start_n; n < end_n; n++) {
            // Dequantize in chunks for better cache locality
            for (int k = 0; k < K; k += QK8_0) {
                custom_dequantize_row_q8_0(src[(n * K + k) / QK8_0].qs, temp + k, QK8_0, 
                                         ggml_compute_fp16_to_fp32(src[(n * K + k) / QK8_0].d));
            }
            
            custom_quantize_row_q8_0(temp, temp_qs, K, delta[n]);
            
            // Copy to destination with optimized memory access pattern
            for (int k = 0; k < K; k += subK) {
                const int dst_offset = (((n / subN) * K_remain + (k / subK)) * subN + (n % subN)) * subK;
                memcpy(dst + dst_offset, temp_qs + k, subK * sizeof(int8_t));
            }
        }
    };
    
    // Launch threads
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back(worker, t);
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

/*
    * @brief convert norm layout to perf layout
    * column major -> row major
    * norm layout (GGML) (col): [K, M] (index: [k + m * K])
    * perf layout (RKNN) (row): [K/subK, M, subK] (index: [(i * M + m) * subK + j] == [i][m][j])
    * for RK3588 we align to 16, INT8 subK=16 

    * since the weights quantize in the K dimention, 
    * here we also quantize in the K dimention
    
        int8:
        native layout: (K / 16, M, 16)
            [M0K0, M0K1,  ..., M0K15,
            M1K0, M1K1, ..., M1K15,
            ...
            M(m-1)K0, M(m-1)K1, ..., M(m-1)K15,

            M0K16, M0K17, ..., M0K31,
            M1K16, M1K17, ..., M1K31,
            ...
            M(m-1)K16, M(m-1)K17, ..., M(m-1)K31,
            ...
            M(m-1)K(k-15), M(m-1)K(k-14), ..., M(m-1)K(k-1)]

            

    * and I will group [M0K0, M0K1, ..., M0K15, M0K16, ..., M0K(k-1)] as a block, 
    * to find MmKk in quant, it's in Block floor((m * K + k) / 32), index (k % 32)

*/
void norm_layout_to_perf_layout_AC_q8_0(const float * src, int8_t * dst, float16 * delta, int32_t M, int32_t K, int32_t subK = 16) {
    // only supports q8
    // QK8_0 = 32;

    //TODO: parallelize it 

    // int num_blocks = (K * M) / QK8_0;
    // int outer_size = (K + subK - 1) / subK; // K1

    float* temp = new float[QK8_0];
    block_q8_0 temp_q8_0;

    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            int q_index = k % QK8_0;
            temp[q_index] = src[k + m * K];

            if (q_index == QK8_0 - 1) {
                // temp is full 
                quantize_row_q8_0_ref(temp, &temp_q8_0, QK8_0);

                for (int s = 0; s < QK8_0 / subK; s++) {
                    memcpy(dst + ((k / subK + s) * M + m) * subK, temp_q8_0.qs + s * subK, sizeof(int8_t) * subK);
                }
                delta[q_index] = temp_q8_0.d;
            }
            
        }
    }

    delete [] temp;
}

/*
    quantize in the K dimention
    that is, to find MmKk in quant, it's in Block m, index k

    * for calibrating the indices: 
    * normal layout (human readable): A[m][k], its delta is delta[m]
    * perf layout (RKNN): it's located in A_perf[floor(k/subK)][m][k % subK]
*/
void norm_layout_to_perf_layout_A_custom_q8_0(const float * src, int8_t * dst, float * delta, int32_t M, int32_t K, int32_t subK = 16) {
    // only supports q8
    // QK8_0 = 32;

    // Parallelize over M dimension with OpenMP
    // #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        // Allocate temp buffer per thread to avoid contention
        int8_t* temp_qs = new int8_t[K];
        
        // Quantize the row
        custom_quantize_row_q8_0(src + m * K, temp_qs, K, delta[m]);

        // Copy quantized data to destination with better memory access pattern
        // Process subK blocks in parallel for better cache utilization
        #pragma omp parallel num_threads(omp_threads) for schedule(static)
        for (int k = 0; k < K; k += subK) {
            int dst_offset = (k / subK) * M * subK + m * subK;
            memcpy(dst + dst_offset, temp_qs + k, subK * sizeof(int8_t));
        }
        
        delete[] temp_qs;
    }
}

template <typename T>
void B_memcpy_multithread(T *dst, T *src, int32_t row, int32_t col){
    // replace the memcpy with multi thread
    // memcpy(dst, src, row * col * sizeof(T)); 
    std::vector<std::thread> threads;

    // int num_threads = std::thread::hardware_concurrency();
    //TODO: config num_threads
    int num_threads = omp_threads;
    
    auto worker = [&](int thread_id) {
        int elements_per_thread = (row * col + num_threads - 1) / num_threads;
        int start_element = thread_id * elements_per_thread;
        int end_element = std::min((thread_id + 1) * elements_per_thread, row * col);
        
        if (start_element < end_element) {
            memcpy((T*)(dst + start_element), (T*)(src + start_element), (end_element - start_element) * sizeof(T));
        }
    };
    
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }

}


/**
     * @brief convert perf to norm layout
     * perf layout: [K1, M, subK]
     * norm layout: [M,K]
     *
     */
template <typename Ti, typename To>
void perf_layout_to_norm_layout_C(Ti * src, To * dst, int32_t M, int32_t N, int32_t K_remain, int32_t subK) {
    for (int i = 0; i < K_remain; i++) {
        for (int j = 0; j < subK; j++) {
            for (int m = 0; m < M; m++) {      
                //TODO: parallelize it
                dst[m * N + i * subK + j] = src[i * M * subK + m * subK + j];

            }
        }
    }
}

void perf_layout_to_norm_layout_C_custom_q8_0(const int32_t * src, float * dst, const float* delta_a, const float* delta_b, int32_t M, int32_t N, int32_t N_remain, int32_t subN) {
    for (int i = 0; i < N_remain; i++) {
        for (int j = 0; j < subN; j++) {
            for (int m = 0; m < M; m++) {
                
                //TODO: parallelize it
                dst[m * N + i * subN + j] = src[i * M * subN + m * subN + j] * delta_a[m] * delta_b[i * subN + j];

            }
        }
    }
}

/*
    for calibrating the indices: 
    GGML human readable (shape [M]): C[m][n] -> dst[m][col_start + n] -> dst[m * ori_N + col_start + n]
    RKNN perf layout (shape [N_remain = N / subN, M, subN]): 
        C_perf[floor(n/subN)][m][n % subN] -> src[floor(n/subN) * M * subN + m * subN + n % subN]
*/
void perf_layout_to_ggml_layout_C(const float * src, float * dst, int32_t M, int32_t N, int32_t ori_N, int32_t col_start, int32_t subN = 4) {
    // unparallelized code: 
    // for (int m = 0; m < M; m++) {
    //     for (int n = 0; n < N; n += subN) {
    //         // dst[m * ori_N + col_start + n] = src[n / subN * M * subN + m * subN + n % subN];
    //         memcpy(dst + m * ori_N + col_start + n, src + (n / subN) * M * subN + m * subN, subN * sizeof(float));
    //     }
    // }

    // Optimize for M=1 case
    if (M == 1) {
        float* dst_base = dst + col_start;
        const float* src_base = src;
        
        // Process in chunks for better cache locality
        const int32_t chunk_size = 64;
        for (int32_t n = 0; n < N; n += chunk_size) {
            const int32_t end_n = std::min(n + chunk_size, N);
            
            for (int32_t i = n; i < end_n; i += subN) {
                const int32_t block_idx = i / subN;
                const int32_t src_offset = block_idx * subN;
                memcpy(dst_base + i, src_base + src_offset, subN * sizeof(float));
            }
        }
        return;
    }
    
    // Multi-threaded version for M > 1
    const int32_t num_threads = std::min(M, (int32_t)omp_threads);
    std::vector<std::thread> threads;
    
    auto worker = [&](int32_t thread_id) {
        const int32_t rows_per_thread = (M + num_threads - 1) / num_threads;
        const int32_t start_m = thread_id * rows_per_thread;
        const int32_t end_m = std::min(start_m + rows_per_thread, M);
        
        for (int32_t m = start_m; m < end_m; m++) {
            float* dst_row_base = dst + m * ori_N + col_start;
            
            // Process in chunks for better cache locality
            const int32_t chunk_size = 64;
            for (int32_t n = 0; n < N; n += chunk_size) {
                const int32_t end_n = std::min(n + chunk_size, N);
                
                for (int32_t i = n; i < end_n; i += subN) {
                    const int32_t block_idx = i / subN;
                    const int32_t src_offset = block_idx * M * subN + m * subN;
                    memcpy(dst_row_base + i, src + src_offset, subN * sizeof(float));
                }
            }
        }
    };
    
    for (int32_t t = 0; t < num_threads; t++) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

void perf_layout_to_ggml_layout_C_q8_0(const int32_t * src, float * dst, const float* delta_a, const float* delta_b, int32_t M, int32_t N, int32_t ori_N, int32_t col_start, int32_t subN = 4) {
    // unparallelized code: 
    // for (int m = 0; m < M; m++) {
    //     for (int n = 0; n < N; n += 1) {
    //         dst[m * ori_N + col_start + n] = float(src[n / subN * M * subN + m * subN + n % subN]) * delta_a[m] * delta_b[n];
    //         // printf("dst[%d][%d] = %f\n", m, n, dst[m * ori_N + col_start + n]);
    //     }
    // }


    // Optimize for M=1 decode case
    if (M == 1) {
        const float delta_a_val = delta_a[0];
        const int32_t* src_base = src;
        float* dst_base = dst + col_start;
        
        // Process in chunks for better cache locality
        const int32_t chunk_size = 64;
        for (int32_t n = 0; n < N; n += chunk_size) {
            const int32_t end_n = std::min(n + chunk_size, N);
            
            for (int32_t i = n; i < end_n; i++) {
                const int32_t block_idx = i / subN;
                const int32_t offset_in_block = i % subN;
                const int32_t src_idx = block_idx * subN + offset_in_block;
                
                dst_base[i] = float(src_base[src_idx]) * delta_a_val * delta_b[i];
            }
        }
        return;
    }
    
    // Multi-threaded version for M > 1
    const int32_t num_threads = std::min(M, (int32_t)omp_threads);
    std::vector<std::thread> threads;
    
    auto worker = [&](int32_t thread_id) {
        const int32_t rows_per_thread = (M + num_threads - 1) / num_threads;
        const int32_t start_m = thread_id * rows_per_thread;
        const int32_t end_m = std::min(start_m + rows_per_thread, M);
        
        for (int32_t m = start_m; m < end_m; m++) {
            const float delta_a_val = delta_a[m];
            float* dst_row_base = dst + m * ori_N + col_start;
            
            // Process in chunks for better cache locality
            const int32_t chunk_size = 64;
            for (int32_t n = 0; n < N; n += chunk_size) {
                const int32_t end_n = std::min(n + chunk_size, N);
                
                for (int32_t i = n; i < end_n; i++) {
                    const int32_t block_idx = i / subN;
                    const int32_t offset_in_block = i % subN;
                    const int32_t src_idx = block_idx * M * subN + m * subN + offset_in_block;
                    
                    dst_row_base[i] = float(src[src_idx]) * delta_a_val * delta_b[i];
                }
            }
        }
    };
    
    for (int32_t t = 0; t < num_threads; t++) {
        threads.emplace_back(worker, t);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

//MARK: GGML API

bool ggml_rk_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor);
// prototypes
rknn_tensor_type ggml_type_to_rknn_type(ggml_type type);
rknn_tensor_type rknpu2_matmul_type_to_rknn_type_input(rknn_matmul_type type);
rknn_tensor_type rknpu2_matmul_input_type_to_output_type(rknn_tensor_type type);
rknn_matmul_type rknpu2_matmul_type_from_rknn_type(rknn_tensor_type type);
const char* rknpu2_tensor_type_to_string(rknn_tensor_type type);
size_t get_matmul_input_type_size(rknn_matmul_type type);
// void dequantize_row_q8_0(const block_q8_0 * x, float * y, int64_t k) ;


struct ggml_backend_rknn_context {
    int rknn_threads = 1;
    rknn_timing_helper* timer;
    json *rknn_config_ptr = nullptr;
    int ggml_threads = 1;
};


// rknn tensor type -> rknn matmul type
rknn_matmul_type rknpu2_matmul_type_from_rknn_type(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT16:
            return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        case RKNN_TENSOR_INT8:
            return RKNN_INT8_MM_INT8_TO_INT32;
        case RKNN_TENSOR_INT4:
            return RKNN_INT4_MM_INT4_TO_INT16;
        default:
            GGML_ASSERT(0);
    }
}

// rknn tensor type -> string
const char* rknpu2_tensor_type_to_string(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT32:
            return "FLOAT32";
        case RKNN_TENSOR_FLOAT16:
            return "FLOAT16";
        case RKNN_TENSOR_INT8:
            return "INT8";
        case RKNN_TENSOR_INT16:
            return "INT16";
        case RKNN_TENSOR_INT32:
            return "INT32";
        case RKNN_TENSOR_UINT8:
            return "UINT8";
        case RKNN_TENSOR_UINT16:
            return "UINT16";
        default:
            GGML_ASSERT(0);
    }
}

// ggml type -> rknn tensor type
rknn_tensor_type ggml_type_to_rknn_type(ggml_type type) {
    switch(type) {
        case GGML_TYPE_F32:
            return RKNN_TENSOR_FLOAT32;
        case GGML_TYPE_F16:
            return RKNN_TENSOR_FLOAT16;
        case GGML_TYPE_I8:
            return RKNN_TENSOR_INT8;
        case GGML_TYPE_Q8_0:
            return RKNN_TENSOR_INT8;
        case GGML_TYPE_I16:
            return RKNN_TENSOR_INT16;
        default:
            GGML_ASSERT(0);
    }
}

// rknn_matmul_type -> rknn_tensor_type
rknn_tensor_type rknpu2_matmul_type_to_rknn_type_input(rknn_matmul_type type)
{
    switch(type) {
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return RKNN_TENSOR_FLOAT16;
        case RKNN_INT8_MM_INT8_TO_INT32:
            return RKNN_TENSOR_INT8;
        case RKNN_INT4_MM_INT4_TO_INT16:
            return RKNN_TENSOR_INT4;
        default:
            GGML_ASSERT(0);
    }
}

// rknn_tensor_type -> rknn_tensor_type
rknn_tensor_type rknpu2_matmul_input_type_to_output_type(rknn_tensor_type type)
{
    switch(type) {
        case RKNN_TENSOR_FLOAT16:
            return RKNN_TENSOR_FLOAT32;
        case RKNN_TENSOR_INT8:
            return RKNN_TENSOR_INT32;
        case RKNN_TENSOR_INT4:
            return RKNN_TENSOR_INT16;
        default:
            GGML_ASSERT(0);
    }
}

static inline struct timespec * timespec_sub(const struct timespec *ts_a, const struct timespec *ts_b, struct timespec * ts_out){
    ts_out->tv_sec = ts_a->tv_sec - ts_b->tv_sec;
    ts_out->tv_nsec = ts_a->tv_nsec - ts_b->tv_nsec;
    if (ts_out->tv_nsec < 0) {
        ts_out->tv_sec--;
        ts_out->tv_nsec += 1000000000;
    }
    return ts_out;
}

static inline unsigned long long timespec_ns(const struct timespec * ts){
    return (unsigned long long)ts->tv_sec * 1000000000ull + (unsigned long long)ts->tv_nsec;
}

static ggml_status ggml_backend_rknn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // GGML_LOG("rknn graph compute!!!!!!!!, cgraph->n_nodes: %d\n", cgraph->n_nodes);
    
    for (int i = 0; i < cgraph->n_nodes; i++) {
        timing_debug_printf("rknn graph compute node: %d, node->name: %s\n", i, cgraph->nodes[i]->name);
        ggml_tensor * node = cgraph->nodes[i];

        if (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE) {
            continue;
        }

        struct timespec start_compute_forward;
        clock_gettime(CLOCK_MONOTONIC, &start_compute_forward);
        timing_debug_printf("rknn graph compute node: %d, node->name: %s, node->op: %s, start_time: %llu\n", i, node->name, ggml_op_name(node->op), timespec_ns(&start_compute_forward));
        bool ok = ggml_rk_compute_forward(backend, node);
        if (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
        }
        struct timespec end_compute_forward;
        clock_gettime(CLOCK_MONOTONIC, &end_compute_forward);

        // printf("Node %d: %s (%s) start_time: %llu, end_time: %llu, compute time: %llu ns \n", 
        //     i, 
        //     node->name, 
        //     ggml_op_name(node->op), 
        //     timespec_ns(&start_compute_forward), 
        //     timespec_ns(&end_compute_forward),
        //     timespec_ns(timespec_sub(&end_compute_forward, &start_compute_forward, &end_compute_forward)) 
        // );


        GGML_ASSERT(ok);


    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_rknn2_free(ggml_backend_t backend) {
    //ggml_backend_rknn_context * ctx = (ggml_backend_rknn_context *) backend->context;
    // if(ctx != nullptr) delete ctx;

    for(int i = 0 ; i < matmul_parts_AC_count; i++){
        ggml_rknpu_matmul_part_AC *part_AC = &matmul_parts_AC[i];
        if (part_AC->is_using == true) {
            continue;
        }
        rknn_destroy_mem(part_AC->ctx, part_AC->A);
        rknn_destroy_mem(part_AC->ctx, part_AC->C);
        //TODO: prefill ctx not destroyed
        // rknn_matmul_destroy(part_AC->ctx);
    }

    for(int i = 0 ; i < matmul_parts_B_count; i++){
        ggml_rknpu_matmul_part_B *part_B = &matmul_parts_B[i];
        rknn_destroy_mem(part_B->ctx, part_B->B);
        rknn_matmul_destroy(part_B->ctx);
    }

    delete backend;
}

static const char * ggml_backend_rknn_name(ggml_backend_t backend) {
    return "RKNN";

    UNUSED(backend);
}
static void ggml_backend_rknn_free(ggml_backend_t backend) {
    ggml_rknn2_free(backend);

    GGML_UNUSED(backend);
}

void ggml_backend_rknn_set_n_threads(ggml_backend_t backend_rknn, int n_threads) {
    // Check for NULL pointer first
    if (backend_rknn == NULL || sizeof(ggml_backend_t) == 0) {
        fprintf(stderr, "ggml-rknn: ERROR - backend_rknn is NULL\n");
        return;
    }
    
    timing_debug_printf("ggml-rknn: ggml_backend_rknn_set_n_threads: backend_ptr=%p n_threads=%d\n", (void*)backend_rknn, n_threads);
    GGML_ASSERT(ggml_backend_is_rknn(backend_rknn));
    ggml_backend_rknn_context * ctx = (ggml_backend_rknn_context *) backend_rknn->context;
    // ctx->n_threads                  = n_threads;
    // if (n_threads > 3) { ctx->n_threads = 3;} 
    
    // TODO: hardcode 6 threads
    ctx->rknn_threads = 3;
    ctx->ggml_threads = n_threads;

    // ctx->rknn_config = &local_rknn_config;
    // ctx->timer = rknn_timing_helper();
    ctx->timer = &local_timer;
    local_timer.rknn_threads = ctx->rknn_threads;
    // printf("n_threads: %d\n", n_threads);
}

void ggml_backend_rknn_set_is_prefill(bool is_prefill) {
    g_rknn_is_prefill.store(is_prefill, std::memory_order_release);
    g_rknn_prefill_explicitly_set.store(true, std::memory_order_release);
    // Don't log during warmup — it's misleading since warmup offloads both.
#ifdef GGML_RKNN_DEBUG
    if (!g_rknn_is_warmup.load(std::memory_order_relaxed)) {
        printf("ggml-rknn: phase set to %s\n", is_prefill ? "PREFILL" : "DECODE");
    }
#endif
}

void ggml_backend_rknn_set_warmup(bool warmup) {
    g_rknn_is_warmup.store(warmup, std::memory_order_release);

    if (!warmup && !g_warmup_done) {
        // Warmup just ended — freeze the match caches into fast hash-sets.
        // After this point supports_op will use O(1) set lookups instead of
        // regex matching or unordered_map cache probes.
        frozen_decode_offload_set.clear();
        frozen_prefill_offload_set.clear();

        for (const auto &kv : offload_match_cache) {
            if (kv.second) {
                frozen_decode_offload_set.insert(kv.first);
            }
        }
        for (const auto &kv : prefill_offload_match_cache) {
            if (kv.second) {
                frozen_prefill_offload_set.insert(kv.first);
            }
        }

        g_warmup_done = true;
        printf("ggml-rknn: warmup END — frozen decode offload set: %zu nodes, prefill offload set: %zu nodes\n",
               frozen_decode_offload_set.size(), frozen_prefill_offload_set.size());
    } else {
        printf("ggml-rknn: warmup %s\n", warmup ? "START" : "END");
    }
}

static ggml_backend_i ggml_backend_rknn_i = {
    /* .get_name                = */ ggml_backend_rknn_name,
    /* .free                    = */ ggml_backend_rknn_free,
    /* .set_tensor_async        = */ NULL,  /* ggml_backend_opencl_set_tensor_async */
    /* .get_tensor_async        = */ NULL,  /* ggml_backend_opencl_get_tensor_async */
    /* .cpy_tensor_async        = */ NULL,  /* ggml_backend_opencl_cpy_tensor_async */
    /* .synchronize             = */ NULL,  /* ggml_backend_opencl_synchronize */
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rknn_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};
static int ggml_backend_rknn_n_devices = 1;
static const char * ggml_backend_rknn_reg_get_name(ggml_backend_reg_t reg) {
    return "RKNN";

    GGML_UNUSED(reg);
}
static size_t ggml_backend_rknn_reg_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_rknn_n_devices;

    GGML_UNUSED(reg);
}
static const char * ggml_backend_rknn_device_get_name(ggml_backend_dev_t dev) {
    return "RKNN";
    GGML_UNUSED(dev);
}
static const char * ggml_backend_rknn_device_get_description(ggml_backend_dev_t dev) {
    #if defined(GGML_BLAS_USE_ACCELERATE)
        return "Accelerate";
    #elif defined(GGML_BLAS_USE_MKL)
        return "MKL";
    #elif defined(GGML_BLAS_USE_BLIS)
        return "BLIS";
    #elif defined(GGML_BLAS_USE_NVPL)
        return "NVPL";
    #elif defined(OPENBLAS_VERSION)
        return "OpenBLAS";
    #else
        return "RKNN";
    #endif

    GGML_UNUSED(dev);
}
static void ggml_backend_rknn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // TODO
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        *free = info.freeram;
        *total = info.totalram;
        *free = *free * info.mem_unit;
        *total = *total * info.mem_unit;
    } else {
        std::cout<< "sysinfo failed" << "\n";
    }

    GGML_UNUSED(dev);
}
static enum ggml_backend_dev_type ggml_backend_rknn_device_get_type(ggml_backend_dev_t dev) {
    // ACCEL (not GPU) so that CPU weight repacking (aarch64 NEON) stays enabled.
    // GPU type causes make_cpu_buft_list to skip extra buffer types, which
    // disables optimized weight layouts and slows CPU matmul by ~2-3x.
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}
static void ggml_backend_rknn_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rknn_device_get_name(dev);
    props->description = ggml_backend_rknn_device_get_description(dev);
    props->type        = ggml_backend_rknn_device_get_type(dev);
    ggml_backend_rknn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = { // check src/llama-context.cpp:276 
        /* .async                 = */ true,
        /* .host_buffer           = */ true,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ true,
    };
}
static ggml_backend_t ggml_backend_rknn_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_rknn_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}
static ggml_backend_buffer_type_t ggml_backend_rknn_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}
static ggml_backend_buffer_t ggml_backend_rknn_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

//MARK: SUPPORTS OP
static bool ggml_backend_rknn_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {

    switch (op->op) {
        case GGML_OP_MUL_MAT:
        {
            // Use a reference to avoid deep-copying the JSON on every supports_op call.
            const json& rknn_config = dev->context
                ? *((ggml_backend_rknn_context *)dev->context)->rknn_config_ptr
                : local_rknn_config;

            if (op->ne[0] == 0 || op->ne[1] == 0){
                return false;
            }

            // printf("ggml-rknn: supports_op: %s, %d, %d, %d, %d\n", op->name, op->op, op->ne[1], op->src[0]->ne[0], op->ne[0]);

            if(!rknn_config.value("npu_prefill", false) && !rknn_config.value("npu_decode", false)){
                return false;
            }

            // timing_debug_printf("ggml-rknn: supports_op: %s, %s, (%d,%d,%d)\n", op->name, ggml_op_name(op->op), op->ne[1], op->src[0]->ne[0], op->ne[0]);
            // printf("%s, %s, (%d*%d*%d)\n", op->name, ggml_op_name(op->op), op->ne[1], op->src[0]->ne[0], op->ne[0]);

            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            const struct ggml_tensor * dst = op;
            const int64_t ne00 = src0->ne[0]; // k
            const int64_t ne01 = src0->ne[1]; // n
            const int64_t ne10 = src1->ne[0]; // k
            const int64_t ne11 = src1->ne[1]; // m
            const int64_t ne0 = dst->ne[0]; // n
            const int64_t ne1 = dst->ne[1]; // m

            // Determine prefill/decode phase for kernel/weight initialization.
            //
            // During warmup sched_reserve: ne1>1 reliably distinguishes
            // PP graph (prefill) from TG graph (decode).
            //
            // During actual inference (graph_compute → sched_alloc_graph):
            // the caller has set g_rknn_is_prefill via set_is_prefill(),
            // which is authoritative.  We must use it because some ops
            // like result_output have ne1=1 even during prefill.
            //
            // Fallback: if set_is_prefill was never called (shouldn't
            // happen after warmup), use ne1>1 heuristic.
            bool is_warmup = g_rknn_is_warmup.load(std::memory_order_relaxed);
            bool is_prefill;
            if (is_warmup) {
                // During warmup reserve, ne1 is reliable (PP graph vs TG graph)
                is_prefill = (ne1 > 1);
            } else if (g_rknn_prefill_explicitly_set.load(std::memory_order_relaxed)) {
                // After warmup, use the authoritative flag from graph_compute
                is_prefill = g_rknn_is_prefill.load(std::memory_order_relaxed);
            } else {
                // Fallback (should not happen in normal flow)
                is_prefill = (ne1 > DECODE_MAX_M);
            }

            //TODO: when input text is long, GGML sends zero tensors
            bool result = false;

            if(dst->type != GGML_TYPE_F32){
                return false;
            }

            // if (strcmp(op->name, "result_output") == 0) {
            //TODO: configurable
            // if (strstr(op->name, "ffn_") || strcmp(op->name, "result_output") == 0) {
            bool to_offload = false;
            bool have_loaded = false;

            // ---- Fast path: post-warmup, use frozen hash-set lookup ----
            // After warmup is done the set of offloadable op names is fixed.
            // A single unordered_set::count() replaces all regex / cache work.
            if (g_warmup_done && !is_warmup) {
                const std::string name_str(op->name);

                if (!is_prefill) {
                    have_loaded = user_loaded_nodes_set.count(name_str) > 0;
                }

                const auto &frozen_set = is_prefill ? frozen_prefill_offload_set : frozen_decode_offload_set;
                to_offload = frozen_set.count(name_str) > 0;

                timing_debug_printf("ggml-rknn: [fast] %s %s: to_offload=%d have_loaded=%d (%ld * %ld * %ld)\n",
                                    is_prefill ? "prefill" : "decode", op->name,
                                    to_offload, have_loaded, ne1, ne00, ne0);

                if (!g_prefill_release_done && !is_prefill) {
                    // release the prefill nodes with M > MAX_M_DECODE 
                    timing_debug_printf("ggml-rknn: release prefill nodes with M > %d\n", DECODE_MAX_M);

                    for (int i = 0; i < matmul_parts_AC_count; i++) {
                        ggml_rknpu_matmul_part_AC *part = &matmul_parts_AC[i];
                        if (part->M > DECODE_MAX_M && part->is_using == false) {
                            // WARNING: set is_using to true to mark no more usage of this node 
                            // cannot release matmul_parts_AC[i] directly, since it's a pointer to the array
                            timing_debug_printf("ggml-rknn: release prefill node AC: (%d, %d, %d)\n", part->M, part->K, part->N);
                            rknn_destroy_mem(part->ctx, part->A);
                            rknn_destroy_mem(part->ctx, part->C);
                            // rknn_matmul_destroy(part->ctx);
                            part->is_using = true; 
                        }
                    }

                    g_prefill_release_done = true;
                    timing_debug_printf("ggml-rknn: release prefill done\n");
                }
            } else {
                // ---- Slow path: warmup — do regex matching & populate caches ----
                // Only check loaded_nodes during decode (and warmup-for-decode).
                // loaded_nodes entries are decode-side resources; checking them
                // during prefill has no functional effect (result is always false)
                // but causes the scheduler to enter extra code paths and visit the
                // RKNN backend during sched_reserve for TG graphs, which can
                // create unnecessary graph splits and trigger re-allocations.
                if (!is_prefill) {
                    // Use user_loaded_nodes_set (the original config whitelist) so that
                    // nodes dynamically inserted into loaded_nodes_set during warmup
                    // (e.g. prefill-only nodes) do NOT accidentally appear as "loaded"
                    // in the decode phase.
                    if (user_loaded_nodes_set.find(std::string(op->name)) != user_loaded_nodes_set.end()) {
                        have_loaded = true;
                        timing_debug_printf("ggml-rknn: loaded node: %s (%ld * %ld * %ld)\n", op->name, ne1, ne00, ne0);
                    }
                } 

                // Select pattern list and cache based on prefill vs decode.
                // During warmup: check BOTH pattern lists to offload all ops and
                // initialize all kernels (prefill + decode).
                const auto &patterns = is_prefill ? compiled_prefill_offload_patterns : compiled_offload_patterns;
                auto &match_cache = is_prefill ? prefill_offload_match_cache : offload_match_cache;
                const char *mode_str = is_warmup ? "warmup" : (is_prefill ? "prefill" : "decode");

                // Always run pattern matching regardless of have_loaded.
                // to_offload only means the weight is in NPU memory (possibly
                // from warmup); it does NOT mean the op should run on NPU in
                // every phase.  The offload patterns decide that.
                if (!patterns.empty()) {
                    std::string name_str(op->name);
                    auto cache_it = match_cache.find(name_str);
                    if (cache_it != match_cache.end()) {
                        to_offload = cache_it->second;
                        if (to_offload) {
                            timing_debug_printf("ggml-rknn: %s offload node (cached): %s (%ld * %ld * %ld)\n", mode_str, op->name, ne1, ne00, ne0);
                        }
                    } else {
                        for (const auto &pattern : patterns) {
                            if (std::regex_match(name_str, pattern)) {
                                to_offload = true;
                                timing_debug_printf("ggml-rknn: %s offload node: %s (%ld * %ld * %ld)\n", mode_str, op->name, ne1, ne00, ne0);
                                break;
                            }
                        }
                        match_cache[name_str] = to_offload;
                    }
                }

                // During warmup, also check the OTHER pattern list to offload
                // ops that belong to the other phase (prefill-only or decode-only).
                if (is_warmup && !to_offload && !have_loaded) {
                    const auto &other_patterns = is_prefill ? compiled_offload_patterns : compiled_prefill_offload_patterns;
                    auto &other_cache = is_prefill ? offload_match_cache : prefill_offload_match_cache;
                    if (!other_patterns.empty()) {
                        std::string name_str(op->name);
                        auto cache_it = other_cache.find(name_str);
                        if (cache_it != other_cache.end()) {
                            to_offload = cache_it->second;
                        } else {
                            for (const auto &pattern : other_patterns) {
                                if (std::regex_match(name_str, pattern)) {
                                    to_offload = true;
                                    timing_debug_printf("ggml-rknn: warmup offload (other phase) node: %s (%ld * %ld * %ld)\n", op->name, ne1, ne00, ne0);
                                    break;
                                }
                            }
                            other_cache[name_str] = to_offload;
                        }
                    }
                }
            } // end slow path

            if (to_offload || have_loaded) {
                if (ne00 > MAX_K) {
                    // fprintf(stderr, "ggml-rknn: too large K: %d for %s (%d * %d * %d), don't support offloading! \n", ne00, op->name, ne1, ne00, ne0);
                    return false;
                }
                if (ne1 > MAX_M_WARMUP){
                    //TODO: support warmup bs 512?
                    return false;
                }

                if (is_prefill) {
                    // Prefill: only offload pattern-matched nodes (loaded_nodes
                    // are decode-side entries and should not run in prefill).
                    result = rknn_config["npu_prefill"] && to_offload;
                } else if (is_warmup) {
                    // During warmup, always offload matched ops regardless of
                    // npu_prefill/npu_decode config — the goal is to initialize.
                    result = true;
                } else {
                    // Decode: use user_loaded_nodes_set as the authoritative whitelist
                    // if it is non-empty (i.e. the user explicitly configured it).
                    // Only nodes listed there will be offloaded; pattern-matched nodes
                    // (to_offload) are intentionally excluded so that prefill-only or
                    // warmup-loaded nodes are not accidentally offloaded in decode.
                    // If user_loaded_nodes_set is empty, fall back to offload_nodes patterns.
                    if (!user_loaded_nodes_set.empty()) {
                        result = rknn_config["npu_decode"] && have_loaded;
                    } else {
                        result = rknn_config["npu_decode"] && to_offload;
                    }
                }

                if (to_offload && result) {
                    uint64_t type_size = 2;
                    if (op->src[1]->type == GGML_TYPE_F16) {
                        type_size = 2;
                    } else if (op->src[1]->type == GGML_TYPE_Q8_0) {
                        type_size = 1;
                    }

                    // Permanent memory: only the weight matrix B (cached in src_w->extra).
                    // A (activation) and C (output) are temporary per-inference buffers;
                    // during decode M=1 so they are negligible. We must NOT include the
                    // prefill M dimension here or the estimate explodes for large sequences.
                    uint64_t weight_bytes = (uint64_t)ne01 * ne00 * type_size;

                    timing_debug_printf("ggml-rknn: weight_bytes for %s: (%ld * %ld * %ld) weight=%lu bytes\n", op->name, ne1, ne00, ne0, weight_bytes);
                    timing_debug_printf("ggml-rknn: rknpu2_allocated_bytes: %lu bytes\n", rknpu2_allocated_bytes);

                    // Only account for memory if the node has NOT been loaded yet.
                    // supports_op is called multiple times (warmup reservation + actual
                    // prefill + each decode step), so we must avoid double-counting.
                    // NOTE: have_loaded is only set during decode (from user_loaded_nodes_set);
                    // during prefill it is always false, so we also check loaded_nodes_set
                    // to guard against duplicate push_back into the JSON array.
                    if (!have_loaded) {
                        static std::mutex rknpu2_allocated_bytes_mutex;
                        {
                            std::lock_guard<std::mutex> lock(rknpu2_allocated_bytes_mutex);

                            // Skip if already dynamically added (e.g. during an
                            // earlier prefill supports_op call in the same warmup).
                            if (loaded_nodes_set.count(std::string(op->name)) > 0) {
                                timing_debug_printf("ggml-rknn: node already in loaded_nodes_set, skipping duplicate: %s\n", op->name);
                            } else {
                            uint64_t projected = rknpu2_allocated_bytes + weight_bytes;

                            if (projected < MAX_RKNN_MEMORY) {
                                loaded_nodes_set.insert(std::string(op->name));
                                local_rknn_config["loaded_nodes"].push_back(op->name);
                                rknpu2_allocated_bytes = projected;

                                timing_debug_printf("ggml-rknn: to offload -> loaded node: %s (%ld * %ld * %ld)\n", op->name, ne1, ne00, ne0);
                                #ifdef RKNN_MATMUL_DEBUG_TIMING_DETAILS
                                    std::string loaded_nodes_str = "{";
                                    for (auto it = loaded_nodes_set.begin(); it != loaded_nodes_set.end(); ++it) {
                                        if (it != loaded_nodes_set.begin()) {loaded_nodes_str += ", ";}
                                        loaded_nodes_str += "\"" + *it + "\"";
                                    }
                                    loaded_nodes_str += "}";
                                    timing_debug_printf("ggml-rknn: loaded_nodes set: %s\n", loaded_nodes_str.c_str());
                                #endif
                            }
                            else {
                                fprintf(stderr, "ggml-rknn: requires too much memory when loading \"%s\" (%ld * %ld * %ld), resting offload_nodes! \n", op->name, ne1, ne00, ne0);
                                fprintf(stderr, "ggml-rknn: weight_bytes: %lu, already allocated: %lu, max memory: %llu\n", weight_bytes, rknpu2_allocated_bytes, MAX_RKNN_MEMORY);
                                fprintf(stderr, "ggml-rknn: local_rknn_config: %s\n", local_rknn_config.dump().c_str());
                                fprintf(stderr, "ggml-rknn: rknn_config: %s\n", rknn_config.dump().c_str());

                                local_rknn_config["offload_nodes"].clear();
                                local_rknn_config["prefill_offload_nodes"].clear();
                                compiled_offload_patterns.clear();
                                offload_match_cache.clear();
                                compiled_prefill_offload_patterns.clear();
                                prefill_offload_match_cache.clear();

                                result = false;
                            }
                            } // end else (not-duplicate)
                        }
                    } else {
                        timing_debug_printf("ggml-rknn: node already loaded, skipping memory accounting: %s\n", op->name);
                    }
                }
            }
            // printf("ggml_backend_rknn_device_supports_op: %s, %d, %d, %d, %d\n", op->name, result, ne01, ne00, ne11); // n, k, m in rknn's notation
           return result;

        }

        default:
            return false;

    }

    GGML_UNUSED(dev);
}
static bool ggml_backend_rknn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(dev);
}
static const struct ggml_backend_device_i ggml_backend_rknn_device_i = {
    /* .get_name             = */ ggml_backend_rknn_device_get_name,
    /* .get_description      = */ ggml_backend_rknn_device_get_description,
    /* .get_memory           = */ ggml_backend_rknn_device_get_memory,
    /* .get_type             = */ ggml_backend_rknn_device_get_type,
    /* .get_props            = */ ggml_backend_rknn_device_get_props,
    /* .init_backend         = */ ggml_backend_rknn_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_rknn_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_rknn_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_rknn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rknn_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};
static ggml_backend_dev_t ggml_backend_rknn_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_rknn_device = {
        /* .iface   = */ ggml_backend_rknn_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,

    };

    // return &g_ggml_backend_rknn_device;
    return &ggml_backend_rknn_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static struct ggml_backend_reg_i ggml_backend_rknn_reg_i = {
    /* .get_name         = */ ggml_backend_rknn_reg_get_name,
    /* .device_count     = */ ggml_backend_rknn_reg_device_count,
    /* .device_get       = */ ggml_backend_rknn_reg_device_get,
    /* .get_proc_address = */ ggml_backend_rknn_get_proc_address,
};

static void * ggml_backend_rknn_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *)ggml_backend_rknn_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return NULL;
}

ggml_backend_reg_t ggml_backend_rknn_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    if (!initialized) {
         reg = ggml_backend_reg {
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface   = */ ggml_backend_rknn_reg_i,
            /* .context = */ NULL,
         };

        initialized = true;
    }

    return &reg;
}
static const char * ggml_backend_rknn_buffer_type_get_name(ggml_backend_buffer_type_t buffer_type) {
    return "RKNN";
    GGML_UNUSED(buffer_type);
}

static ggml_guid_t ggml_backend_rknn_guid() {
    //c9bdb702-4936-4212-af35-a287d8c02920
    static ggml_guid guid = { 0xc9, 0xbd, 0xb7, 0x02, 0x49, 0x36, 0x42, 0x12, 0xaf, 0x35, 0xa2, 0x87, 0xd8, 0xc9, 0x29, 0x20 };
    return &guid;
}

bool ggml_backend_is_rknn(ggml_backend_t backend){
    return backend != NULL && ggml_guid_matches(backend -> guid, ggml_backend_rknn_guid());
}

// MARK: rknn INIT

ggml_backend_t ggml_backend_rknn_init(void) {
    // printf("@ggml-rknn.cpp\n");
    printf("ggml-rknn: start rknn init!\n");
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_rknn_reg(), 0);
    printf("ggml-rknn: register the rknn!\n");
    ggml_backend_rknn_context * context = (ggml_backend_rknn_context *) malloc(sizeof(ggml_backend_rknn_context));
    printf("ggml-rknn: creating the backend!\n");
    

    ggml_backend_t backend = new ggml_backend {
        /* .guid      = */  ggml_backend_rknn_guid(),
        /* .interface = */  ggml_backend_rknn_i,
        /* .device    = */  dev,
        /* .context   = */  context
    };
    printf("ggml-rknn: done creating rknn backend!\n");

    context->rknn_config_ptr = &local_rknn_config;
    init_rknn_config(context->rknn_config_ptr);

    return backend;
}


size_t get_matmul_input_type_size(rknn_matmul_type type){
    switch(type){
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return sizeof(rknpu2::float16);
        case RKNN_INT8_MM_INT8_TO_INT32:
            return sizeof(int8_t);
        default:
            GGML_ASSERT(0);
    }
    return 0;
}

int get_subN(rknn_matmul_type type){
    switch(type){
        case RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32:
            return 16;
        case RKNN_INT8_MM_INT8_TO_INT32:
            return 32;
        default: 
            GGML_ASSERT(0);
            return 0;
    }
}

// MARK: Matmul sub
extern "C" __attribute__((visibility("default"))) 
void compute_submat_mul( // matrix A row
                        int core_mask_bits,
                        ggml_tensor *src_i,
                        ggml_tensor *src_w,
                        ggml_tensor *dst,
                        void *A_data,
                        void *B_data,
                        void *B_data_delta, // only for Q8
                        int64_t col_start,
                        int64_t col_end,
                        int thread_idx,
                        rknn_matmul_type type,
                        rknn_timing_helper *timer_p
                    )
{
    int64_t ori_N = src_w->ne[1];
    int64_t M = src_i->ne[1];
    int64_t K = src_w->ne[0];
    int64_t N = col_end - col_start;

    if (N == 0) {
        printf("ggml-rknn: n is 0 for %s:%d, %ld, %ld\n", dst->name, thread_idx, col_start, col_end);
        return;
    }

    //TODO: size not padded
    // this dataflow makes me bad
    mat_info mat_A;
    mat_info mat_B;
    if (type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32){
        mat_A = mat_info(M, K, FLOAT16, A_data, true);
        mat_B = mat_info(K, N, FLOAT16, B_data, false);
    } 
    else if (type == RKNN_INT8_MM_INT8_TO_INT32){
        mat_A = mat_info(M, K, INT8, A_data, true);
        mat_B = mat_info(K, N, INT8, B_data, false);
    }
    else{
        GGML_ASSERT(0);
    }
    
    int64_t A_pad_row_00 = mat_A.pad_row; // m
    int64_t A_pad_col_00 = mat_A.pad_col; // k
    int64_t B_pad_row_00 = mat_B.pad_row; // k
    int64_t B_pad_col_00 = mat_B.pad_col; // n

    GGML_ASSERT(B_pad_row_00 == A_pad_col_00);

    // size_t A_size = mat_A.pad_size;
    // size_t B_size = mat_B.pad_size;

    int ret = 0;

    std::vector<int> durations(10, 0);

    // Encode decode + prefill ac_layout_perf into a bitmask: bit0=decode, bit1=prefill.
    const bool decode_ac_perf  = match_ac_layout_perf_node(dst->name);
    const bool prefill_ac_perf = match_ac_layout_perf_prefill_node(dst->name);
    const int  ac_layout_perf_flags = (decode_ac_perf ? 1 : 0) | (prefill_ac_perf ? 2 : 0);

    timing_debug_printf("ggml-rknn: ac_layout_perf flags: %s decode=%d prefill=%d\n",
                        dst->name, decode_ac_perf, prefill_ac_perf);

    ggml_rknpu_matmul_pair rkpair;
    TIMEIT(
        rkpair = create_matmul_pair(M, K, N, type, thread_idx, dst->name, core_mask_bits, ac_layout_perf_flags);
    , &durations[0]);
    timing_debug_printf("ggml-rknn: create_matmul_pair %s:%d time: %d us\n", dst->name, thread_idx, durations[0]);

    ggml_rknpu_matmul_part_AC *part_AC = rkpair.part_AC;
    ggml_rknpu_matmul_part_B *part_B = rkpair.part_B;

    //TODO: this part_B offset is not good for multithread
    
    GGML_ASSERT(part_AC != NULL);
    GGML_ASSERT(part_B != NULL);
        
    part_AC->is_using = true;

    float * A_delta = NULL; // only for q8_0

    // copy A to layout, memcpy B to kernel
    // part_AC->ac_layout_perf reflects the actual AC_layout baked into this variant's ctx.
    {
      if (part_AC->ac_layout_perf) {
        if (type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32){
            TIMEIT(
                ggml_layout_to_perf_layout_A((float *)A_data, (rknpu2::float16 *)part_AC->A->virt_addr, A_pad_row_00, A_pad_col_00, part_AC->io_attr.A.dims[2]);
            , &durations[1]);

            if(!part_B->B_is_copied){
                TIMEIT(
                    B_memcpy_multithread((float16*)part_B->B->virt_addr, (float16*)mat_B.pad_data, B_pad_row_00, B_pad_col_00);
                , &durations[2]);
            }
        } else if (type == RKNN_INT8_MM_INT8_TO_INT32){
            A_delta = new float[A_pad_row_00];

            TIMEIT(
                norm_layout_to_perf_layout_A_custom_q8_0((float *)A_data, (int8_t *)part_AC->A->virt_addr, A_delta, A_pad_row_00, A_pad_col_00, part_AC->io_attr.A.dims[2])
            , &durations[1]);

            if(!part_B->B_is_copied){
                TIMEIT(
                    B_memcpy_multithread((int8_t*)part_B->B->virt_addr, (int8_t*)mat_B.pad_data, B_pad_row_00, B_pad_col_00);
                , &durations[2]);

                #if GGML_RKNPU2_USE_OUTSIDE_ALLOC
                dma_sync_cpu_to_device(part_B->B->fd);
                #endif
            }
        }

        timing_debug_printf("ggml-rknn: copy A to perf layout, memcpy B to kernel time: %d %d us\n", durations[1], durations[2]);
      } else {
        if (type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32){
            TIMEIT({
                // NORM layout: A is [M, K] row-major FP16
                // GGML column-major src1[K,M] and RKNN row-major A[M,K] share the same
                // memory layout: element (m,k) is at offset m*K+k in both cases.
                const float * src_f32 = (const float *)A_data;
                uint16_t * dst_f16 = (uint16_t *)part_AC->A->virt_addr;
                const int64_t total = M * K;
                int64_t i = 0;
#ifdef __ARM_NEON
                for (; i + 7 < total; i += 8) {
                    float32x4_t lo = vld1q_f32(src_f32 + i);
                    float32x4_t hi = vld1q_f32(src_f32 + i + 4);
                    float16x8_t f16 = vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi));
                    vst1q_u16(dst_f16 + i, vreinterpretq_u16_f16(f16));
                }
#endif
                for (; i < total; i++) {
                    dst_f16[i] = GGML_FP32_TO_FP16(src_f32[i]);
                }
            }, &durations[1]);

            if(!part_B->B_is_copied){
                TIMEIT(
                    B_memcpy_multithread((float16*)part_B->B->virt_addr, (float16*)mat_B.pad_data, B_pad_row_00, B_pad_col_00);
                , &durations[2]);
            }
        } else if (type == RKNN_INT8_MM_INT8_TO_INT32){
            A_delta = new float[A_pad_row_00];

            TIMEIT({
                // NORM layout: A is [M, K] row-major INT8
                // Quantize each row: find max abs, compute scale, quantize
                const float * src_f32 = (const float *)A_data;
                int8_t * dst_i8 = (int8_t *)part_AC->A->virt_addr;
                for (int64_t m = 0; m < M; m++) {
                    const float * row = src_f32 + m * K;
                    float amax = 0.0f;
                    for (int64_t k = 0; k < K; k++) {
                        amax = std::max(amax, std::abs(row[k]));
                    }
                    float scale = amax / 127.0f;
                    A_delta[m] = scale;
                    float iscale = (scale == 0.0f) ? 0.0f : 1.0f / scale;
                    int8_t * dst_row = dst_i8 + m * K;
                    for (int64_t k = 0; k < K; k++) {
                        dst_row[k] = (int8_t)roundf(row[k] * iscale);
                    }
                }
            }, &durations[1]);

            if(!part_B->B_is_copied){
                TIMEIT(
                    B_memcpy_multithread((int8_t*)part_B->B->virt_addr, (int8_t*)mat_B.pad_data, B_pad_row_00, B_pad_col_00);
                , &durations[2]);

                #if GGML_RKNPU2_USE_OUTSIDE_ALLOC
                dma_sync_cpu_to_device(part_B->B->fd);
                #endif
            }
        }

        timing_debug_printf("ggml-rknn: copy A to norm layout, memcpy B to kernel time: %d %d us\n", durations[1], durations[2]);
      }
    }

    // set io to rknn
    {

        TIMEIT(
            rknn_matmul_set_io_mem(part_AC->ctx, part_AC->A, &(part_AC->io_attr.A));
        , &durations[3]);

        if(!(part_B->B_is_copied))
        {
            // if b is not copied, or if it's prefill case
            TIMEIT(
                rknn_matmul_set_io_mem(part_B->ctx, part_B->B, &(part_B->io_attr.B));
            , &durations[4]);

            //TODO: bad readability of B_is_copied
            part_B->B_is_copied = true;
        }

        if(part_AC->prefill)
        {
            TIMEIT(
                rknn_matmul_set_io_mem(part_AC->ctx, part_B->B, &(part_AC->io_attr.B));
            , &durations[4]);
        }

        TIMEIT(
            rknn_matmul_set_io_mem(part_AC->ctx, part_AC->C, &(part_AC->io_attr.C));
        , &durations[5]);

        // rknn_mem_sync(part_AC->ctx, part_AC->A, RKNN_MEMORY_SYNC_TO_DEVICE);
        // rknn_mem_sync(part_AC->ctx, part_B->B, RKNN_MEMORY_SYNC_TO_DEVICE);

        timing_debug_printf("ggml-rknn: set io time: %d us\n", durations[3] + durations[4] + durations[5]);
        timing_debug_printf("ggml-rknn: set io time A: %d us, B: %d us, C: %d us\n", durations[3], durations[4], durations[5]);
    }

    // submit matrix mult to rknn
    {

        TIMEIT(
            ret = rknn_matmul_run(part_AC->ctx);
        , &durations[6]);

        if (ret != 0) {
            printf("ggml-rknn: rknn_matmul_run failed for %s:%d\n", part_B->name, part_AC->thread_idx);
        }
        // rknn_mem_sync(part_AC->ctx, part_AC->C, RKNN_MEMORY_SYNC_FROM_DEVICE);

        timing_debug_printf("ggml-rknn: rknn_matmul_run duration: %d us\n", durations[6]);

    }
    
    // #ifdef RKNN_MATMUL_DEBUG
    // {        
        // if (strcmp(dst->name, "ffn_out-0") == 0) {
            // printf("kernel %d dump: \n", thread_idx);
            // dump_matmul_tensor(part_AC->A, &part_AC->io_attr.A);
            // dump_matmul_tensor(part_B->B, &part_AC->io_attr.B);
            // dump_matmul_tensor(part_AC->C, &part_AC->io_attr.C);
        // }
    // }
    // #endif 

    // rknn result back to ggml 
    {
      if (part_AC->ac_layout_perf) {
        if (type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32){
            TIMEIT({
                perf_layout_to_ggml_layout_C((const float *)part_AC->C->virt_addr, (float *)dst->data, M, N, ori_N, col_start);
                part_AC->is_using = false;
            }, &durations[8]);

        } else if (type == RKNN_INT8_MM_INT8_TO_INT32){
            TIMEIT({
                perf_layout_to_ggml_layout_C_q8_0((const int32_t *)part_AC->C->virt_addr, (float *)dst->data, A_delta, (const float *)B_data_delta, M, N, ori_N, col_start);
                part_AC->is_using = false;
            }, &durations[8]);
        }
      } else {
        if (type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32){
            TIMEIT({
                // C is [M, N] row-major FP32 → copy each row to dst at col_start offset
                const float * c_data = (const float *)part_AC->C->virt_addr;
                float * dst_data = (float *)dst->data;
                for (int64_t m = 0; m < M; m++) {
                    memcpy(dst_data + m * ori_N + col_start,
                           c_data + m * N,
                           N * sizeof(float));
                }
                part_AC->is_using = false;
            }, &durations[8]);

        } else if (type == RKNN_INT8_MM_INT8_TO_INT32){
            TIMEIT({
                // C is [M, N] row-major INT32 → dequantize and copy
                const int32_t * c_data = (const int32_t *)part_AC->C->virt_addr;
                float * dst_data = (float *)dst->data;
                for (int64_t m = 0; m < M; m++) {
                    float scale_a = A_delta[m];
                    float * dst_row = dst_data + m * ori_N + col_start;
                    const int32_t * c_row = c_data + m * N;
                    for (int64_t n = 0; n < N; n++) {
                        dst_row[n] = (float)c_row[n] * scale_a * ((float *)B_data_delta)[n];
                    }
                }
                part_AC->is_using = false;
            }, &durations[8]);
        }
      }
    }
    {

        // free(norm_layout_C);

        timing_debug_printf("ggml-rknn: write back to column-major GGML dst duration: %d us\n", durations[8]);

        //TODO: refactor free prefill case 

        // TIMEIT(
        //     if (part_AC->prefill) {
        //         timing_debug_printf("ggml-rknn: free prefill case %s:%d (%ld,%ld,%ld)\n", part_AC->name, part_AC->thread_idx, A_pad_row_00, A_pad_col_00, B_pad_col_00);

        //         rknn_destroy_mem(part_B->ctx, part_AC->A);
        //         rknn_destroy_mem(part_B->ctx, part_AC->C);
        //         rknn_matmul_destroy(part_B->ctx);
        //         part_AC->A = nullptr;
        //         part_AC->C = nullptr;
        //     }
        // , &durations[9]);

        // timing_debug_printf("ggml-rknn: free prefill case duration: %d us\n", durations[9]);

        #ifdef RKNN_MATMUL_DEBUG_TIMING_INFO
            timer_p->find_kernel_time += durations[0];
            timer_p->prepare_data_time_A += durations[1];
            timer_p->memcpy_to_kernel_time += durations[1];
            timer_p->memcpy_to_kernel_time += durations[2];
            timer_p->set_io_time_A += durations[3];
            timer_p->set_io_time_B += durations[4];
            timer_p->set_io_time_C += durations[5];
            timer_p->set_io_time += durations[3] + durations[4] + durations[5];
            timer_p->run_time += durations[6];
            timer_p->prepare_data_time_C += durations[7];
            timer_p->memcpy_to_result_time += durations[8];
            timer_p->free_pointer_time += durations[9];
        #endif
    }

    //TODO: need to release padding pointer is padding used

    // printf("ggml-rknn: compute_submat_mul done for %s:%d, (%d,%d,%d)\n", dst->name, thread_idx, A_pad_row_00, A_pad_col_00, B_pad_col_00);
}


//MARK: DEBUG HELPER
// 一维矩阵乘法函数
float* matrixMultiply_vector_omp(const float16* A, const float* B, int N, int K, int M) {
    // A: [K, N] B: [K, M] -> result: [N, M]
    // in column major order
    float* result = new float[M * N];
    std::fill(result, result + M * N, float(0));

    // Parallelize outer loops using OpenMP
    #pragma omp parallel num_threads(omp_threads) for collapse(2) schedule(dynamic)
    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            const float16* A_col = A + n * K;
            const float* B_col = B + m * K;
            
            // Optimized case for float16 * float -> float
            float sum = 0.0f;
            
            // Vectorize inner loop and use loop unrolling for better performance
            int k = 0;
            // Process 4 elements at a time for float16 (better SIMD utilization)
            for (; k <= K - 4; k += 4) {
                sum += static_cast<float>(A_col[k]) * B_col[k] +
                       static_cast<float>(A_col[k+1]) * B_col[k+1] +
                       static_cast<float>(A_col[k+2]) * B_col[k+2] +
                       static_cast<float>(A_col[k+3]) * B_col[k+3];
            }
            // Handle remaining elements
            for (; k < K; ++k) {
                sum += static_cast<float>(A_col[k]) * B_col[k];
            }
            
            result[n + m * N] = sum;
        }
    }

    return result;
}

float* matrixMultiply_vector(const float16* A, const float* B, int N, int K, int M) {
    float* result = new float[M * N];
    std::fill(result, result + M * N, float(0));
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                // if (m == 0 && n <= 1) {
                //     printf("A[%d * %d + %d] = %f, B[%d * %d + %d] = %f\n", n, K, k, (float) A[n * K + k], m, K, k, (float) B[m * K + k]);
                // }
                sum += (float) A[n * K + k] * (float) B[m * K + k];
            }
            result[n + m * N] = sum;
        }
    }
    return result;
}

template <typename T>
void col_to_row_transpose(T * matrix, T * dest_matrix, int rows, int cols) {
    // assume the input matrix is in column major order
    // access matrix[i][j] as matrix[i + j * rows]
    // access dest_matrix[i][j] as dest_matrix[i * cols + j]
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dest_matrix[i * cols + j] = matrix[i + j * rows];
        }
    }
}

template <typename T>
float arraysCosineSimilarity_v(const T* arr1, const T* arr2, size_t size) {
    // 计算点积
    double dotProduct = 0.0;
    for (size_t i = 0; i < size; ++i) {
        dotProduct += arr1[i] * arr2[i];
    }

    // 计算向量范数
    double normA = 0.0;
    double normB = 0.0;
    for (size_t i = 0; i < size; ++i) {
        normA += std::pow(arr1[i], 2);
        normB += std::pow(arr2[i], 2);
    }

    if (normA == 0.0 || normB == 0.0) {
        return 0.0;
    }

    return dotProduct / (std::sqrt(normA) * std::sqrt(normB));
}

// MARK: ggml_rk_mul_mat
// TODO: Unify the meaning of variable names, m, k, n to rknn's definition

/*  
    in this function we do the transform between GGML (column major) and RKNN (row major)
    take ffn_up_0 as example of matrix length :
        (1, 2048)(intermediate) * (2048, 8192)(weight) -> (8192, 1)(output)

    GGML node: (column major)
    src0: weight (F16) (2048, 8192) (K, N)
    src1: intermediate (F32) (2048, 1) (K, M)
    dst:  output (F32) (8192, 1) (N, M)

    RKNN requires (row major) F16 * F16 -> F32 (M, K, N) = (1, 2048, 8192)
    in norm layout it should be:  
    A: intermediate ((F32) -> (F16)) (1, 2048) (M, K)
    B: weight (F16) (2048, 8192) (K, N)
    C: output (F32) (1, 8192) (M, N)
*/
extern "C" __attribute__((visibility("default"))) 
void ggml_rk_mul_mat(int *pt_npu_core_mask, ggml_backend_t backend, ggml_tensor * src_w, ggml_tensor * src_i, ggml_tensor * dst, rknn_matmul_type inference_type) {
    std::vector<std::thread> threads;
    int rknn_threads = (int)((ggml_backend_rknn_context * )backend->context)->rknn_threads;
    rknn_timing_helper *timer_p = ((ggml_backend_rknn_context *)backend->context)->timer;
    threads.reserve(rknn_threads);
    
    int N = src_w->ne[1];
    int K = src_w->ne[0];
    // int M = src_i->ne[1];

    // convert F32 to F16
    // also do the column major to row major
    /*
        accessing ggml matrix src_i[k0, m0] -> src_i[k0 + m0 * k]
        accessing rknn matrix A_data[m0, k0] -> A_data[m0 * k + k0]
    */

    int duration_b = 0;
    
    // /*
    //     accessing ggml matrix src_w[k0, n0] -> src_w[k0 + n0 * k]
    //     accessing rknn matrix B_data[k0, n0] -> B_data[k0 * n + n0]
    // */

    
    void * B_native_data;
    void * B_data_delta;

    TIMEIT(
        if (inference_type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32) {
            if (src_w->extra == NULL) {
                B_native_data = malloc(K * N * sizeof(rknpu2::float16));
                ggml_layout_to_native_layout_B_multithreads((rknpu2::float16 *)src_w->data, (rknpu2::float16 *)B_native_data, K, N, 16, 32);
                
                src_w->extra = B_native_data;
            }else{
                B_native_data = src_w->extra;
            }
        }
        else if (inference_type == RKNN_INT8_MM_INT8_TO_INT32) {
            if (src_w->extra == NULL) {
                // quantize by dimention K 
                B_native_data = malloc(K * N * sizeof(int8_t));
                B_data_delta = malloc(N * sizeof(float));
                // ggml_layout_to_native_layout_B_custom_q8_0((block_q8_0 *)src_w->data, (int8_t *)B_native_data, (float *)B_data_delta, k, n, 32, 32);
                ggml_layout_to_native_layout_B_custom_q8_0_multithread((block_q8_0 *)src_w->data, (int8_t *)B_native_data, (float *)B_data_delta, K, N, 32, 32);

                // Store both pointers in a struct or pair
                void** extra_data = (void**)malloc(2 * sizeof(void*));
                extra_data[0] = B_native_data;
                extra_data[1] = B_data_delta;
                src_w->extra = extra_data;
            }else{
                B_native_data = ((void **)src_w->extra)[0];
                B_data_delta = ((void **)src_w->extra)[1];
            }
        }
        
    , &duration_b);

    #ifdef RKNN_MATMUL_DEBUG_TIMING_INFO
    timer_p->prepare_data_time_B += duration_b;
    #endif
    
    timing_debug_printf("ggml-rknn: copying weights B duration: %d us\n", duration_b);

    // Use popcount(core_mask) threads so each thread maps to exactly one unique NPU core.
    // e.g. core_mask=0x6 (CORE_1+CORE_2) -> 2 threads: thread0->CORE_1, thread1->CORE_2
    int threads_number = std::min(rknn_threads, __builtin_popcount((unsigned)*pt_npu_core_mask));
    npu_core_debug_printf("[NPU_CORE] ggml_rk_mul_mat: rknn_threads=%d core_mask=0x%x (%s) -> effective threads=%d\n",
                          rknn_threads, *pt_npu_core_mask, core_mask_to_str(*pt_npu_core_mask), threads_number);

    for(int t = 0; t < threads_number; t++){
        //TODO: assert sub_n is divisible by 16, consider the padding later 
        // devide b for multi thread
        int subN = get_subN(inference_type);

        GGML_ASSERT(N % subN == 0);
        int n_quotient = N / subN;
        int64_t col_start = t * n_quotient / threads_number * subN;
        int64_t col_end = (t + 1) * n_quotient / threads_number * subN;
        if (col_end > N){
            col_end = N;
        }

        // shape of B_native is [N/subN, K/subK, subN, subK]
        // so we need to offset the data
        // col_start is in normal layout view
        void * B_compute_data = (void *)((char *)B_native_data + col_start * K * get_matmul_input_type_size(inference_type));

        void * B_compute_data_delta = (void *)((float *)B_data_delta + col_start);

        void * A_compute_data = src_i->data; // src_i is in F32 
        // void * B_compute_data = B_data;

        // run the thread;
        threads.emplace_back([A_compute_data, B_compute_data, B_compute_data_delta, dst, col_start, col_end, t, inference_type, src_i, src_w, timer_p, pt_npu_core_mask](){
            compute_submat_mul(*pt_npu_core_mask, src_i, src_w, dst, A_compute_data, B_compute_data, B_compute_data_delta, col_start, col_end, t, inference_type, timer_p);
        });
    }
    // #ifdef RKNN_MATMUL_DEBUG
    // printf("layer_name: %s\n", dst->name);
    // #endif 

    for (auto & th : threads) {
        th.join();
    }

}

// typedef void (*ggml_rk_func_t)(ggml_backend_t backend, ggml_tensor * src0, ggml_tensor * src1, ggml_tensor * dst, rknn_matmul_type type);

bool ggml_rk_compute_forward(ggml_backend_t backend, struct ggml_tensor * tensor) {
    // ggml_rk_func_t func = nullptr;

    ggml_tensor * src0 = tensor->src[0];
    ggml_tensor * src1 = tensor->src[1];

    if(tensor->op != GGML_OP_MUL_MAT){
        return false;
    }

    // func = ggml_rk_mul_mat;
    int npu_core_mask = get_op_npu_core_mask(tensor->name, local_rknn_config["npu_core_mask"].get<int>());
    npu_core_debug_printf("[NPU_CORE] ggml_rk_compute_forward: tensor=%s M=%lld K=%lld N=%lld core_mask=0x%x (%s)\n",
                          tensor->name, (long long)tensor->ne[1], (long long)src0->ne[0], (long long)tensor->ne[0], npu_core_mask, core_mask_to_str(npu_core_mask));
    
    rknn_matmul_type matmul_type;
    matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    //in ggml graph (column-major) src0 (F16/Q8_0 weight), src1 (F32 intermediate), dst (F32 output)
    // in rknn (row-major) left A (F16 intermediate, perf) and right B(F16 weight, native)
    if(src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32){
        matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    }else if(src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32){
        matmul_type = RKNN_INT8_MM_INT8_TO_INT32;
    }

    // GGML_LOG("ggml_rk_compute_forward: tensor->name: %s, tensor->op: %s, src0->name: %s, src1->name: %s\n", tensor->name, ggml_op_name(tensor->op), src0->name, src1->name);


    timing_debug_printf("ggml-rknn: starting matmul for %s size %ld x %ld x %ld\n", tensor->name, src0->ne[0], src0->ne[1], src1->ne[1]);

    TIMEIT(
        ggml_rk_mul_mat(&npu_core_mask, backend, tensor->src[0], tensor->src[1], tensor, matmul_type)
    , &((ggml_backend_rknn_context *)backend->context)->timer->total_run_time);

    // printf("ggml-rknn: processed tensor: %s, (%d,%d,%d) num_npu_cores: %d\n", tensor->name, tensor->ne[1], src0->ne[0], tensor->ne[0], num_npu_cores);

    // Log current phase.
#ifdef RKNN_MATMUL_DEBUG_TIMING_INFO
    {
        const char *phase_str;
        if (g_rknn_is_warmup.load(std::memory_order_relaxed)) {
            phase_str = "WARMUP";
        } else if (g_rknn_prefill_explicitly_set.load(std::memory_order_relaxed)) {
            phase_str = g_rknn_is_prefill.load(std::memory_order_relaxed) ? "PREFILL" : "DECODE";
        } else {
            phase_str = (tensor->ne[1] > 1) ? "PREFILL(ne1)" : "DECODE(ne1)";
        }
    //     printf("ggml-rknn: [%s] %s  M=%lld K=%lld N=%lld  cores=%d\n",
    //            phase_str, tensor->name,
    //            (long long)tensor->ne[1], (long long)src0->ne[0], (long long)tensor->ne[0],
    //            num_npu_cores);
    // }
    }
#endif


    #ifdef RKNN_MATMUL_DEBUG_TIMING_INFO
        if (strstr(tensor->name, "output") != NULL || strcmp(tensor->name, "node_0") == 0){
            ((ggml_backend_rknn_context *)backend->context)->timer->dump_time_usage();
            ((ggml_backend_rknn_context *)backend->context)->timer->clear();
        }
    #endif 

    return true;
}
