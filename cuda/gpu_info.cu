// gpu_info.cu
#include <cuda_runtime.h>
#include <driver_types.h> // Add this include for UUID functions
#include <cstdio>
// ... rest of includes
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

#define CHECK_CUDA(call)                                                       \
    do                                                                         \
    {                                                                          \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
        {                                                                      \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                              \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

static const char *boolStr(int v) { return v ? "Yes" : "No"; }

int main()
{
    int driverVersion = 0, runtimeVersion = 0;
    CHECK_CUDA(cudaDriverGetVersion(&driverVersion));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVersion));

    int deviceCount = 0;
    cudaError_t e = cudaGetDeviceCount(&deviceCount);
    if (e == cudaErrorNoDevice)
    {
        std::cout << "No CUDA-capable GPU detected.\n";
        return 0;
    }
    CHECK_CUDA(e);

    std::cout << "CUDA Driver Version / Runtime Version: "
              << driverVersion / 1000 << "." << (driverVersion % 100) / 10
              << " / "
              << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10
              << "\n";
    std::cout << "Detected CUDA Devices: " << deviceCount << "\n\n";

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        CHECK_CUDA(cudaSetDevice(dev));

        cudaDeviceProp prop{};
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        // PCI Bus ID string
        char busId[64] = {0};
        cudaError_t busRes = cudaDeviceGetPCIBusId(busId, sizeof(busId), dev);

        // Optional: UUID (CUDA 10.0+), LUID (Windows, CUDA 11.2+)
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 10000) && defined(__CUDA_RUNTIME_H__)
        cudaUUID_t uuid{};
        cudaError_t hasUuid = cudaSuccess;

// Check if function exists at runtime
#ifdef cudaDeviceGetUuid
        hasUuid = cudaDeviceGetUuid(&uuid, dev);
#else
        hasUuid = cudaErrorNotSupported;
#endif

        if (hasUuid == cudaSuccess)
        {
            std::cout << "UUID:                             ";
            for (int i = 0; i < 16; ++i)
            {
                std::printf("%02x", static_cast<unsigned>(uuid.bytes[i]));
                if (i == 3 || i == 5 || i == 7 || i == 9)
                    std::printf("-");
            }
            std::printf("\n");
        }
#endif
        // Query current free/total memory on this device
        size_t freeMem = 0, totalMem = 0;
        CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem));

        // Handy function to query many attributes
        auto getAttr = [&](cudaDeviceAttr a) -> int
        {
            int v = 0;
            CHECK_CUDA(cudaDeviceGetAttribute(&v, a, dev));
            return v;
        };

        std::cout << "=============================================\n";
        std::cout << "Device " << dev << ": " << prop.name << "\n";
        std::cout << "=============================================\n";

        std::cout << "Compute Capability:               " << prop.major << "." << prop.minor << "\n";
        std::cout << "Total Global Memory:              " << (prop.totalGlobalMem / (1024.0 * 1024.0)) << " MiB\n";
        std::cout << "Free / Total (runtime):           "
                  << (freeMem / (1024.0 * 1024.0)) << " / "
                  << (totalMem / (1024.0 * 1024.0)) << " MiB\n";
        std::cout << "Multiprocessors (SMs):            " << prop.multiProcessorCount << "\n";
        std::cout << "Max Threads per SM:               " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "Max Threads per Block:            " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Warp Size:                        " << prop.warpSize << "\n";
        std::cout << "Regs per Block:                   " << prop.regsPerBlock << "\n";
#if CUDART_VERSION >= 11000
        std::cout << "Regs per SM:                      " << prop.regsPerMultiprocessor << "\n";
#endif
        std::cout << "Shared Mem per Block:             " << (prop.sharedMemPerBlock / 1024.0) << " KiB\n";
#if CUDART_VERSION >= 11000
        std::cout << "Shared Mem per SM:                " << (prop.sharedMemPerMultiprocessor / 1024.0) << " KiB\n";
#endif
#ifdef cudaDevAttrMaxSharedMemoryPerBlockOptin
        int smemOptin = getAttr(cudaDevAttrMaxSharedMemoryPerBlockOptin);
        if (smemOptin > 0)
            std::cout << "Max Shared Mem per Block (optin): " << (smemOptin / 1024.0) << " KiB\n";
#endif
        std::cout << "Max Block Dim:                    "
                  << prop.maxThreadsDim[0] << " x " << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << "\n";
        std::cout << "Max Grid  Dim:                    "
                  << prop.maxGridSize[0] << " x " << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << "\n";
        std::cout << "Memory Bus Width:                 " << getAttr(cudaDevAttrGlobalMemoryBusWidth) << " bits\n";
        int memClockKHz = getAttr(cudaDevAttrMemoryClockRate);
        std::cout << "Memory Clock Rate:                " << (memClockKHz / 1000.0) << " MHz\n";
        int coreClockKHz = getAttr(cudaDevAttrClockRate);
        std::cout << "Core Clock Rate:                  " << (coreClockKHz / 1000.0) << " MHz\n";
        std::cout << "L2 Cache Size:                    " << getAttr(cudaDevAttrL2CacheSize) / 1024 << " KiB\n";
        std::cout << "ECC Enabled:                      " << boolStr(prop.ECCEnabled) << "\n";
        std::cout << "Concurrent Kernels:               " << boolStr(getAttr(cudaDevAttrConcurrentKernels)) << "\n";
        std::cout << "Concurrent Managed Access:        " << boolStr(getAttr(cudaDevAttrConcurrentManagedAccess)) << "\n";
        std::cout << "Unified Addressing (UVA):         " << boolStr(getAttr(cudaDevAttrUnifiedAddressing)) << "\n";
        std::cout << "Managed Memory:                   " << boolStr(getAttr(cudaDevAttrManagedMemory)) << "\n";
        std::cout << "Cooperative Launch:               " << boolStr(getAttr(cudaDevAttrCooperativeLaunch)) << "\n";
#ifdef cudaDevAttrCooperativeMultiDeviceLaunch
        std::cout << "Cooperative Multi-Device:         " << boolStr(getAttr(cudaDevAttrCooperativeMultiDeviceLaunch)) << "\n";
#else
        std::cout << "Cooperative Multi-Device:         Not available\n";
#endif
        std::cout << "Can Map Host Memory:              " << boolStr(prop.canMapHostMemory) << "\n";
        std::cout << "DMA Engines (async engines):      " << getAttr(cudaDevAttrAsyncEngineCount) << "\n";
        std::cout << "PCI Domain:Bus:Device:            "
                  << std::hex << std::setfill('0') << std::setw(4) << prop.pciDomainID << ":"
                  << std::setw(2) << prop.pciBusID << ":" << std::setw(2) << prop.pciDeviceID
                  << std::dec << "\n";
        if (busRes == cudaSuccess)
        {
            std::cout << "PCI Bus Id (string):              " << busId << "\n";
        }
#if CUDART_VERSION >= 10000
        if (hasUuid == cudaSuccess)
        {
            std::cout << "UUID:                             ";
            for (int i = 0; i < 16; ++i)
            {
                std::printf("%02x", static_cast<unsigned>(uuid.bytes[i]));
                if (i == 3 || i == 5 || i == 7 || i == 9)
                    std::printf("-");
            }
            std::printf("\n");
        }
#endif

        // Selected device attributes (query via cudaDeviceGetAttribute)
        // Selected device attributes (query via cudaDeviceGetAttribute)
        struct AttrItem
        {
            cudaDeviceAttr attr;
            const char *name;
        };

        std::vector<AttrItem> attrs;

        // Add attributes that are definitely available
        attrs.push_back({cudaDevAttrMaxThreadsPerMultiProcessor, "Max Threads / SM"});
        attrs.push_back({cudaDevAttrMaxRegistersPerBlock, "Max Registers / Block"});
        attrs.push_back({cudaDevAttrWarpSize, "Warp Size"});
        attrs.push_back({cudaDevAttrMaxSharedMemoryPerBlock, "Max Shared Mem / Block (bytes)"});
        attrs.push_back({cudaDevAttrMemoryClockRate, "Memory Clock (kHz)"});
        attrs.push_back({cudaDevAttrClockRate, "Core Clock (kHz)"});
        attrs.push_back({cudaDevAttrGlobalMemoryBusWidth, "Memory Bus Width (bits)"});
        attrs.push_back({cudaDevAttrL2CacheSize, "L2 Cache (bytes)"});
        attrs.push_back({cudaDevAttrComputeCapabilityMajor, "CC Major"});
        attrs.push_back({cudaDevAttrComputeCapabilityMinor, "CC Minor"});
        attrs.push_back({cudaDevAttrConcurrentKernels, "Concurrent Kernels"});
        attrs.push_back({cudaDevAttrUnifiedAddressing, "Unified Addressing"});
        attrs.push_back({cudaDevAttrManagedMemory, "Managed Memory"});
        attrs.push_back({cudaDevAttrCooperativeLaunch, "Cooperative Launch"});

// Add version-dependent attributes
#ifdef cudaDevAttrMultiprocessorCount
        attrs.push_back({cudaDevAttrMultiprocessorCount, "SM Count"});
#endif

#ifdef cudaDevAttrMaxBlocksPerMultiProcessor
        attrs.push_back({cudaDevAttrMaxBlocksPerMultiProcessor, "Max Blocks / SM"});
#endif

#ifdef cudaDevAttrMaxRegistersPerMultiprocessor
        attrs.push_back({cudaDevAttrMaxRegistersPerMultiprocessor, "Max Registers / SM"});
#endif

#ifdef cudaDevAttrMaxSharedMemoryPerBlockOptin
        attrs.push_back({cudaDevAttrMaxSharedMemoryPerBlockOptin, "Max Shared Mem / Block (opt-in, attr)"});
#endif

#ifdef cudaDevAttrSharedMemoryPerMultiprocessor
        attrs.push_back({cudaDevAttrSharedMemoryPerMultiprocessor, "Shared Mem / SM (attr)"});
#endif

#ifdef cudaDevAttrConcurrentManagedAccess
        attrs.push_back({cudaDevAttrConcurrentManagedAccess, "Concurrent Managed Access"});
#endif

#ifdef cudaDevAttrPageableMemoryAccess
        attrs.push_back({cudaDevAttrPageableMemoryAccess, "Pageable Memory Access"});
#endif

#ifdef cudaDevAttrPageableMemoryAccessUsesHostPageTables
        attrs.push_back({cudaDevAttrPageableMemoryAccessUsesHostPageTables, "Pageable Access uses Host PageTables"});
#endif

#ifdef cudaDevAttrCooperativeMultiDeviceLaunch
        attrs.push_back({cudaDevAttrCooperativeMultiDeviceLaunch, "Cooperative Multi-Device Launch"});
#endif

#ifdef cudaDevAttrCanUseHostPointerForRegisteredMem
        attrs.push_back({cudaDevAttrCanUseHostPointerForRegisteredMem, "Use Host Pointer for Registered Mem"});
#endif

#ifdef cudaDevAttrHostRegisterSupported
        attrs.push_back({cudaDevAttrHostRegisterSupported, "Host Register Supported"});
#endif

#ifdef cudaDevAttrDirectManagedMemAccessFromHost
        attrs.push_back({cudaDevAttrDirectManagedMemAccessFromHost, "Direct Managed Access from Host"});
#endif

#ifdef cudaDevAttrMemoryPoolsSupported
        attrs.push_back({cudaDevAttrMemoryPoolsSupported, "Memory Pools Supported"});
#endif

#ifdef cudaDevAttrMaxSharedMemoryPerMultiprocessor
        attrs.push_back({cudaDevAttrMaxSharedMemoryPerMultiprocessor, "Max Shared Mem / SM"});
#endif

#ifdef cudaDevAttrClusterLaunch
        attrs.push_back({cudaDevAttrClusterLaunch, "Cluster Launch Supported"});
#endif

#ifdef cudaDevAttrClusterMultiDeviceLaunch
        attrs.push_back({cudaDevAttrClusterMultiDeviceLaunch, "Cluster Multi-Device Launch"});
#endif

        std::cout << "\nDevice Attributes (via cudaDeviceGetAttribute):\n";
        for (const auto &it : attrs)
        {
            int v = 0;
            cudaError_t st = cudaDeviceGetAttribute(&v, it.attr, dev);
            if (st == cudaSuccess)
            {
                // Check if this is a boolean attribute
                bool isBoolAttr = (it.attr == cudaDevAttrConcurrentKernels ||
                                   it.attr == cudaDevAttrUnifiedAddressing ||
                                   it.attr == cudaDevAttrManagedMemory ||
                                   it.attr == cudaDevAttrCooperativeLaunch
#ifdef cudaDevAttrConcurrentManagedAccess
                                   || it.attr == cudaDevAttrConcurrentManagedAccess
#endif
#ifdef cudaDevAttrPageableMemoryAccess
                                   || it.attr == cudaDevAttrPageableMemoryAccess
#endif
#ifdef cudaDevAttrPageableMemoryAccessUsesHostPageTables
                                   || it.attr == cudaDevAttrPageableMemoryAccessUsesHostPageTables
#endif
#ifdef cudaDevAttrCooperativeMultiDeviceLaunch
                                   || it.attr == cudaDevAttrCooperativeMultiDeviceLaunch
#endif
#ifdef cudaDevAttrCanUseHostPointerForRegisteredMem
                                   || it.attr == cudaDevAttrCanUseHostPointerForRegisteredMem
#endif
#ifdef cudaDevAttrHostRegisterSupported
                                   || it.attr == cudaDevAttrHostRegisterSupported
#endif
#ifdef cudaDevAttrDirectManagedMemAccessFromHost
                                   || it.attr == cudaDevAttrDirectManagedMemAccessFromHost
#endif
#ifdef cudaDevAttrMemoryPoolsSupported
                                   || it.attr == cudaDevAttrMemoryPoolsSupported
#endif
#ifdef cudaDevAttrClusterLaunch
                                   || it.attr == cudaDevAttrClusterLaunch
#endif
#ifdef cudaDevAttrClusterMultiDeviceLaunch
                                   || it.attr == cudaDevAttrClusterMultiDeviceLaunch
#endif
                );

                // Format output properly
                std::cout << "  " << std::left << std::setfill(' ') << std::setw(45) << it.name
                          << ": " << (isBoolAttr ? boolStr(v) : std::to_string(v)) << "\n";
            }
        }

        // NICETIES: report default memory pool granularity if supported
#if CUDART_VERSION >= 11020
#ifdef cudaDevAttrMemoryPoolsSupported
        if (getAttr(cudaDevAttrMemoryPoolsSupported))
        {
            cudaMemPool_t pool;
            CHECK_CUDA(cudaDeviceGetDefaultMemPool(&pool, dev));
            size_t gran = 0;
            CHECK_CUDA(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &gran));
            std::cout << "\nDefault MemPool Release Threshold: " << gran << " bytes\n";
            // (There are more mempool attributes; shown is a common one.)
        }
#endif
#endif

        // NOTE: Some clocks/power limits/temps require NVML; this program sticks to CUDA Runtime.
        std::cout << "\n";
    }

    return 0;
}
