// Task #163 stage 1: CUDA/Vulkan GPU-side interop foundation. See
// vulkaninterop.h for the full rationale and mechanism description.

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "vulkaninterop.h"
#include "interop_double_comp_spv.h"
#include "intersect_batch_comp_spv.h"

#define VK_CHECK_GOTO(call, label)                                       \
    do {                                                                 \
        VkResult _r = (call);                                            \
        if (_r != VK_SUCCESS) {                                          \
            fprintf(stderr, "vulkaninterop: Vulkan error %d at %s:%d: %s\n", \
                    _r, __FILE__, __LINE__, #call);                      \
            goto fail;                                                   \
        }                                                                \
    } while (0)

#define CUDA_CHECK_GOTO(call, label)                                     \
    do {                                                                 \
        cudaError_t _c = (call);                                         \
        if (_c != cudaSuccess) {                                         \
            fprintf(stderr, "vulkaninterop: CUDA error %d (%s) at %s:%d: %s\n", \
                    (int)_c, cudaGetErrorString(_c), __FILE__, __LINE__, #call); \
            goto fail;                                                   \
        }                                                                \
    } while (0)

namespace {

uint32_t findMemoryType(VkPhysicalDevice pd, uint32_t typeBits,
                         VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(pd, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return UINT32_MAX;
}

// A simple device-local (or host-visible, if requested) buffer with an
// optional device address -- used below for BLAS/TLAS storage and scratch
// buffers (AS builds, not the CUDA-shared interop buffers, which need
// their own export-chain construction -- see createInteropBuffer).
// Mirrors vulkanrt.cpp's Buffer/createBuffer/destroyBuffer exactly.
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    VkDeviceSize size = 0;
};

bool createBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                   VkDeviceSize size, VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags memProps, bool wantAddress,
                   Buffer* out) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage | (wantAddress ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bci, nullptr, &out->buffer) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateBuffer failed\n");
        return false;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, out->buffer, &req);

    VkMemoryAllocateFlagsInfo flagsInfo{};
    flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flagsInfo.flags = wantAddress ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = wantAddress ? &flagsInfo : nullptr;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(physicalDevice, req.memoryTypeBits, memProps);
    if (mai.memoryTypeIndex == UINT32_MAX) {
        fprintf(stderr, "vulkaninterop: no suitable memory type for buffer\n");
        vkDestroyBuffer(device, out->buffer, nullptr);
        out->buffer = VK_NULL_HANDLE;
        return false;
    }
    if (vkAllocateMemory(device, &mai, nullptr, &out->memory) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkAllocateMemory failed\n");
        vkDestroyBuffer(device, out->buffer, nullptr);
        out->buffer = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(device, out->buffer, out->memory, 0);
    out->size = size;

    if (wantAddress) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = out->buffer;
        out->address = vkGetBufferDeviceAddress(device, &addrInfo);
    }
    return true;
}

void destroyBuffer(VkDevice device, Buffer* b) {
    if (b->buffer) vkDestroyBuffer(device, b->buffer, nullptr);
    if (b->memory) vkFreeMemory(device, b->memory, nullptr);
    *b = Buffer{};
}

bool uploadToBuffer(VkDevice device, const Buffer& buf, const void* data, VkDeviceSize bytes) {
    void* p;
    if (vkMapMemory(device, buf.memory, 0, bytes, 0, &p) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkMapMemory failed\n");
        return false;
    }
    memcpy(p, data, (size_t)bytes);
    vkUnmapMemory(device, buf.memory);
    return true;
}

// Manually-loaded KHR ray-tracing entry points (see vulkanrt.cpp's
// identical RtFunctions/loadRtFunctions -- duplicated here rather than
// shared across bridge libraries, matching this codebase's existing
// per-bridge independence convention).
struct RtFunctions {
    PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
    PFN_vkCreateAccelerationStructureKHR create = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroy = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuild = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getDeviceAddress = nullptr;
};

bool loadRtFunctions(VkDevice device, RtFunctions* fns) {
    fns->getBuildSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
    fns->create = (PFN_vkCreateAccelerationStructureKHR)
        vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
    fns->destroy = (PFN_vkDestroyAccelerationStructureKHR)
        vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR");
    fns->cmdBuild = (PFN_vkCmdBuildAccelerationStructuresKHR)
        vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
    fns->getDeviceAddress = (PFN_vkGetAccelerationStructureDeviceAddressKHR)
        vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR");
    if (!fns->getBuildSizes || !fns->create || !fns->destroy ||
        !fns->cmdBuild || !fns->getDeviceAddress) {
        fprintf(stderr, "vulkaninterop: failed to load one or more KHR acceleration structure entry points\n");
        return false;
    }
    return true;
}

// Builds either a BLAS (triangle geometry) or a TLAS (instance geometry).
// Identical logic to vulkanrt.cpp's buildAccelerationStructure.
bool buildAccelerationStructure(
    VkDevice device, VkPhysicalDevice physicalDevice,
    VkCommandPool cmdPool, VkQueue queue, const RtFunctions& fns,
    VkAccelerationStructureTypeKHR type,
    const VkAccelerationStructureGeometryKHR& geometry,
    uint32_t primitiveCount,
    VkAccelerationStructureKHR* outAS, Buffer* outASBuffer,
    VkDeviceAddress* outASAddress) {

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = type;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fns.getBuildSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                       &buildInfo, &primitiveCount, &sizeInfo);

    if (!createBuffer(device, physicalDevice, sizeInfo.accelerationStructureSize,
                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, outASBuffer)) {
        return false;
    }

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = outASBuffer->buffer;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = type;
    if (fns.create(device, &createInfo, nullptr, outAS) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateAccelerationStructureKHR failed\n");
        return false;
    }

    Buffer scratch;
    if (!createBuffer(device, physicalDevice, sizeInfo.buildScratchSize,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true, &scratch)) {
        return false;
    }

    buildInfo.dstAccelerationStructure = *outAS;
    buildInfo.scratchData.deviceAddress = scratch.address;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfo = &rangeInfo;

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
        destroyBuffer(device, &scratch);
        return false;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    fns.cmdBuild(cmd, 1, &buildInfo, &pRangeInfo);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
    destroyBuffer(device, &scratch);

    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = *outAS;
    *outASAddress = fns.getDeviceAddress(device, &addrInfo);
    return true;
}

// Multi-geometry BLAS variant: object-instancing templates can bundle
// several meshes (e.g. barcelona-pavilion's tree templates have 5-9
// plymesh shapes each -- see pbrt_parser.mojo's ObjectBegin/ObjectEnd), so
// one template needs one BLAS built from MULTIPLE triangle geometries, one
// per mesh, each getting its own geometryIndex (0..geometries.size()-1) so
// a hit can be decoded back to "mesh template_mesh_start[t] + geometryIndex"
// -- see intersect_batch.comp's geometryIndex report and
// vulkaninterop_unpack_results_kernel's decode (gpu.mojo). Otherwise
// identical to the single-geometry buildAccelerationStructure above (same
// scratch/command-buffer/address-query dance), just with geometryCount > 1
// and one VkAccelerationStructureBuildRangeInfoKHR per geometry.
bool buildAccelerationStructureMulti(
    VkDevice device, VkPhysicalDevice physicalDevice,
    VkCommandPool cmdPool, VkQueue queue, const RtFunctions& fns,
    const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
    const std::vector<uint32_t>& primitiveCounts,
    VkAccelerationStructureKHR* outAS, Buffer* outASBuffer,
    VkDeviceAddress* outASAddress) {

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = (uint32_t)geometries.size();
    buildInfo.pGeometries = geometries.data();

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    fns.getBuildSizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                       &buildInfo, primitiveCounts.data(), &sizeInfo);

    if (!createBuffer(device, physicalDevice, sizeInfo.accelerationStructureSize,
                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, false, outASBuffer)) {
        return false;
    }

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = outASBuffer->buffer;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    if (fns.create(device, &createInfo, nullptr, outAS) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateAccelerationStructureKHR (multi) failed\n");
        return false;
    }

    Buffer scratch;
    if (!createBuffer(device, physicalDevice, sizeInfo.buildScratchSize,
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, true, &scratch)) {
        return false;
    }

    buildInfo.dstAccelerationStructure = *outAS;
    buildInfo.scratchData.deviceAddress = scratch.address;

    std::vector<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos(primitiveCounts.size());
    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pRangeInfos(primitiveCounts.size());
    for (size_t i = 0; i < primitiveCounts.size(); i++) {
        rangeInfos[i] = VkAccelerationStructureBuildRangeInfoKHR{};
        rangeInfos[i].primitiveCount = primitiveCounts[i];
        pRangeInfos[i] = &rangeInfos[i];
    }

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device, &cbai, &cmd) != VK_SUCCESS) {
        destroyBuffer(device, &scratch);
        return false;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    fns.cmdBuild(cmd, 1, &buildInfo, pRangeInfos.data());
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
    destroyBuffer(device, &scratch);

    VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
    addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addrInfo.accelerationStructure = *outAS;
    *outASAddress = fns.getDeviceAddress(device, &addrInfo);
    return true;
}

struct Interop {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;

    // The shared buffer: Vulkan-owned memory, exported and imported into CUDA.
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize bufferBytes = 0;
    cudaExternalMemory_t cudaExtMem = nullptr;
    void* cudaPtr = nullptr;

    // Two binary semaphores, each exported and imported into CUDA.
    VkSemaphore semA = VK_NULL_HANDLE; // CUDA signals, Vulkan waits
    VkSemaphore semB = VK_NULL_HANDLE; // Vulkan signals, CUDA waits
    cudaExternalSemaphore_t cudaSemA = nullptr;
    cudaExternalSemaphore_t cudaSemB = nullptr;

    // Compute pipeline: doubles every element (trivial stand-in for real RT
    // work at this stage -- see vulkaninterop.h).
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    // Pre-recorded once (see vulkaninterop_round_trip): the dispatch itself
    // never changes between round trips, only the semaphores it's submitted
    // with (specified at vkQueueSubmit time, not baked into the command
    // buffer), so one recording is reused for every call.
    VkCommandBuffer cmd = VK_NULL_HANDLE;

    int64_t elemCount = 0;
};

// Loaded via vkGetDeviceProcAddr -- VK_KHR_external_memory_fd/
// VK_KHR_external_semaphore_fd entry points aren't linked directly (same
// reasoning as vulkanrt.cpp's RtFunctions: portable across loaders).
struct ExtFns {
    PFN_vkGetMemoryFdKHR getMemoryFd = nullptr;
    PFN_vkGetSemaphoreFdKHR getSemaphoreFd = nullptr;
};

bool loadExtFns(VkDevice device, ExtFns* fns) {
    fns->getMemoryFd = (PFN_vkGetMemoryFdKHR)
        vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    fns->getSemaphoreFd = (PFN_vkGetSemaphoreFdKHR)
        vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
    if (!fns->getMemoryFd || !fns->getSemaphoreFd) {
        fprintf(stderr, "vulkaninterop: failed to load external memory/semaphore fd entry points\n");
        return false;
    }
    return true;
}

// Factors out the "create an exportable Vulkan buffer, get its fd, import
// into CUDA, get the aliasing CUDA pointer" dance (used inline once already
// by vulkaninterop_create for its single test buffer -- left as-is there to
// avoid touching working stage-1 code; this reusable version is for stage
// 2's two RT buffers, rays and results, below).
bool createInteropBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                          const ExtFns& fns, VkDeviceSize bytes,
                          VkBufferUsageFlags usage,
                          VkBuffer* outBuffer, VkDeviceMemory* outMemory,
                          cudaExternalMemory_t* outExtMem, void** outCudaPtr) {
    VkExternalMemoryBufferCreateInfo extBufInfo{};
    extBufInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.pNext = &extBufInfo;
    bci.size = bytes;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bci, nullptr, outBuffer) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateBuffer (interop) failed\n");
        return false;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, *outBuffer, &req);

    VkExportMemoryAllocateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.pNext = &exportInfo;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(physicalDevice, req.memoryTypeBits,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mai.memoryTypeIndex == UINT32_MAX) {
        fprintf(stderr, "vulkaninterop: no suitable device-local memory type for interop buffer\n");
        return false;
    }
    if (vkAllocateMemory(device, &mai, nullptr, outMemory) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkAllocateMemory (interop) failed\n");
        return false;
    }
    if (vkBindBufferMemory(device, *outBuffer, *outMemory, 0) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkBindBufferMemory (interop) failed\n");
        return false;
    }

    int memFd = -1;
    VkMemoryGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    getFdInfo.memory = *outMemory;
    getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (fns.getMemoryFd(device, &getFdInfo, &memFd) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkGetMemoryFdKHR failed\n");
        return false;
    }

    // Ownership of memFd transfers to CUDA on a successful import.
    cudaExternalMemoryHandleDesc memDesc{};
    memset(&memDesc, 0, sizeof(memDesc));
    memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    memDesc.handle.fd = memFd;
    memDesc.size = req.size;
    cudaError_t mc = cudaImportExternalMemory(outExtMem, &memDesc);
    if (mc != cudaSuccess) {
        fprintf(stderr, "vulkaninterop: cudaImportExternalMemory failed: %s\n", cudaGetErrorString(mc));
        return false;
    }

    cudaExternalMemoryBufferDesc bufDesc{};
    memset(&bufDesc, 0, sizeof(bufDesc));
    bufDesc.offset = 0;
    bufDesc.size = bytes;
    cudaError_t bc = cudaExternalMemoryGetMappedBuffer(outCudaPtr, *outExtMem, &bufDesc);
    if (bc != cudaSuccess) {
        fprintf(stderr, "vulkaninterop: cudaExternalMemoryGetMappedBuffer failed: %s\n", cudaGetErrorString(bc));
        return false;
    }
    return true;
}

// Same fd-export/CUDA-import dance as createInteropBuffer, for a binary
// VkSemaphore instead of a buffer.
bool createInteropSemaphore(VkDevice device, const ExtFns& fns,
                             VkSemaphore* outSem, cudaExternalSemaphore_t* outCudaSem) {
    VkExportSemaphoreCreateInfo exportSemInfo{};
    exportSemInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportSemInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &exportSemInfo;
    if (vkCreateSemaphore(device, &sci, nullptr, outSem) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateSemaphore (interop) failed\n");
        return false;
    }

    int semFd = -1;
    VkSemaphoreGetFdInfoKHR getFdInfo{};
    getFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    getFdInfo.semaphore = *outSem;
    getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (fns.getSemaphoreFd(device, &getFdInfo, &semFd) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkGetSemaphoreFdKHR failed\n");
        return false;
    }

    cudaExternalSemaphoreHandleDesc semDesc{};
    memset(&semDesc, 0, sizeof(semDesc));
    semDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    semDesc.handle.fd = semFd;
    cudaError_t sc = cudaImportExternalSemaphore(outCudaSem, &semDesc);
    if (sc != cudaSuccess) {
        fprintf(stderr, "vulkaninterop: cudaImportExternalSemaphore failed: %s\n", cudaGetErrorString(sc));
        return false;
    }
    return true;
}

bool createHeadlessInteropDevice(Interop* it, ExtFns* fns) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "gonzales-vulkaninterop";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &it->instance) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateInstance failed\n");
        return false;
    }

    const char* requiredExts[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    };

    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(it->instance, &count, nullptr);
        if (count == 0) {
            fprintf(stderr, "vulkaninterop: no Vulkan physical devices found\n");
            return false;
        }
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(it->instance, &count, devices.data());
        for (auto pd : devices) {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
            bool hasMemFd = false, hasSemFd = false;
            for (auto& e : exts) {
                if (strcmp(e.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) == 0) hasMemFd = true;
                if (strcmp(e.extensionName, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) == 0) hasSemFd = true;
            }
            if (hasMemFd && hasSemFd) {
                it->physicalDevice = pd;
                break;
            }
        }
        if (!it->physicalDevice) {
            fprintf(stderr, "vulkaninterop: no device supports external memory/semaphore fd extensions\n");
            return false;
        }
    }

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(it->physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(it->physicalDevice, &qfCount, qfs.data());
    uint32_t queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily = i; break; }
    }
    if (queueFamily == UINT32_MAX) {
        fprintf(stderr, "vulkaninterop: no compute-capable queue family\n");
        return false;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 2;
    dci.ppEnabledExtensionNames = requiredExts;
    if (vkCreateDevice(it->physicalDevice, &dci, nullptr, &it->device) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateDevice failed\n");
        return false;
    }
    vkGetDeviceQueue(it->device, queueFamily, 0, &it->queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = queueFamily;
    if (vkCreateCommandPool(it->device, &cpci, nullptr, &it->cmdPool) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateCommandPool failed\n");
        return false;
    }

    return loadExtFns(it->device, fns);
}

// ---------------------------------------------------------------------------
// Stage 2: interop-AND-ray-query-capable scene.
// ---------------------------------------------------------------------------

struct InteropRtScene {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    ExtFns extFns{};
    RtFunctions rtFns{};

    // Sized mesh_count -- one slot per mesh, whether ordinary or template.
    // Ordinary meshes get a real BLAS in blas[i]/blasBufs[i] (identity-
    // transform TLAS instance, same as before instancing existed); template
    // meshes' vertex/index buffers are populated the same way but reused as
    // one GEOMETRY within their template's own multi-geometry BLAS below --
    // blas[i]/blasBufs[i] stay VK_NULL_HANDLE/empty for those indices.
    std::vector<Buffer> blasVertexBufs;
    std::vector<Buffer> blasIndexBufs;
    std::vector<VkAccelerationStructureKHR> blas;
    std::vector<Buffer> blasBufs;

    // Sized template_count -- one multi-geometry BLAS per ObjectBegin/
    // ObjectEnd template, built from that template's mesh range's buffers
    // above (see buildAccelerationStructureMulti).
    std::vector<VkAccelerationStructureKHR> templateBlas;
    std::vector<Buffer> templateBlasBufs;

    Buffer instanceBuf{};
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    Buffer tlasBuf{};

    // The two CUDA-shared buffers, fixed for the scene's lifetime (sized to
    // max_rays at creation time) -- unlike vulkanrt.cpp's vulkanrt_trace_rays,
    // which creates/destroys fresh buffers every call.
    VkBuffer raysBuffer = VK_NULL_HANDLE;
    VkDeviceMemory raysMemory = VK_NULL_HANDLE;
    cudaExternalMemory_t raysCudaExtMem = nullptr;
    void* raysCudaPtr = nullptr;
    VkDeviceSize raysBytes = 0;

    VkBuffer resultsBuffer = VK_NULL_HANDLE;
    VkDeviceMemory resultsMemory = VK_NULL_HANDLE;
    cudaExternalMemory_t resultsCudaExtMem = nullptr;
    void* resultsCudaPtr = nullptr;
    VkDeviceSize resultsBytes = 0;

    VkSemaphore semA = VK_NULL_HANDLE;
    VkSemaphore semB = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cudaSemA = nullptr;
    cudaExternalSemaphore_t cudaSemB = nullptr;

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    int64_t maxRays = 0;

    // Command buffers are cached by ray_count and recorded WITH
    // VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT (see
    // vulkaninterop_rt_trace) so they can be resubmitted every call without
    // per-call allocate/free churn or a CPU-side fence wait -- correctness
    // of resubmitting a possibly-still-pending command buffer relies on
    // that flag plus the GPU-side semaphore ordering, not on the host
    // knowing when a previous submission actually finished. ray_count is
    // typically constant across a whole render (same n_total every bounce)
    // with at most one or two distinct values in practice (e.g. a final
    // partial wavefront batch), so this cache stays tiny.
    std::vector<std::pair<uint32_t, VkCommandBuffer>> cmdCache;
};

bool createHeadlessInteropRtDevice(InteropRtScene* sc) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "gonzales-vulkaninterop-rt";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &sc->instance) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateInstance (rt) failed\n");
        return false;
    }

    const char* requiredExts[] = {
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };
    const int numExts = 5;

    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(sc->instance, &count, nullptr);
        if (count == 0) {
            fprintf(stderr, "vulkaninterop: no Vulkan physical devices found\n");
            return false;
        }
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(sc->instance, &count, devices.data());
        for (auto pd : devices) {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
            bool hasAll[5] = {false, false, false, false, false};
            for (auto& e : exts) {
                for (int i = 0; i < numExts; i++) {
                    if (strcmp(e.extensionName, requiredExts[i]) == 0) hasAll[i] = true;
                }
            }
            bool ok = true;
            for (int i = 0; i < numExts; i++) ok = ok && hasAll[i];
            if (ok) { sc->physicalDevice = pd; break; }
        }
        if (!sc->physicalDevice) {
            fprintf(stderr, "vulkaninterop: no device supports interop + ray-query extensions together\n");
            return false;
        }
    }

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(sc->physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(sc->physicalDevice, &qfCount, qfs.data());
    uint32_t queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily = i; break; }
    }
    if (queueFamily == UINT32_MAX) {
        fprintf(stderr, "vulkaninterop: no compute-capable queue family\n");
        return false;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures{};
    rqFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rqFeatures.rayQuery = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.accelerationStructure = VK_TRUE;
    asFeatures.pNext = &rqFeatures;

    VkPhysicalDeviceVulkan12Features vk12Features{};
    vk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12Features.bufferDeviceAddress = VK_TRUE;
    vk12Features.pNext = &asFeatures;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &vk12Features;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = numExts;
    dci.ppEnabledExtensionNames = requiredExts;
    if (vkCreateDevice(sc->physicalDevice, &dci, nullptr, &sc->device) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateDevice (rt) failed\n");
        return false;
    }
    vkGetDeviceQueue(sc->device, queueFamily, 0, &sc->queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = queueFamily;
    if (vkCreateCommandPool(sc->device, &cpci, nullptr, &sc->cmdPool) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateCommandPool (rt) failed\n");
        return false;
    }

    return loadExtFns(sc->device, &sc->extFns) && loadRtFunctions(sc->device, &sc->rtFns);
}

} // namespace

extern "C" void vulkaninterop_destroy(void* handle) {
    if (!handle) return;
    Interop* it = (Interop*)handle;

    // Required before destroying device-owned objects: ensures the last
    // vulkaninterop_round_trip's GPU work (async, no CPU wait by design)
    // has actually finished. Only runs once at teardown, not in the
    // per-call hot path, so it costs nothing towards the "no CPU sync in
    // the render loop" goal this bridge exists for.
    if (it->device) vkDeviceWaitIdle(it->device);

    if (it->cudaSemA) cudaDestroyExternalSemaphore(it->cudaSemA);
    if (it->cudaSemB) cudaDestroyExternalSemaphore(it->cudaSemB);
    if (it->cudaExtMem) cudaDestroyExternalMemory(it->cudaExtMem);

    if (it->descPool) vkDestroyDescriptorPool(it->device, it->descPool, nullptr);
    if (it->pipeline) vkDestroyPipeline(it->device, it->pipeline, nullptr);
    if (it->pipelineLayout) vkDestroyPipelineLayout(it->device, it->pipelineLayout, nullptr);
    if (it->dsLayout) vkDestroyDescriptorSetLayout(it->device, it->dsLayout, nullptr);
    if (it->shaderModule) vkDestroyShaderModule(it->device, it->shaderModule, nullptr);

    if (it->semA) vkDestroySemaphore(it->device, it->semA, nullptr);
    if (it->semB) vkDestroySemaphore(it->device, it->semB, nullptr);
    if (it->buffer) vkDestroyBuffer(it->device, it->buffer, nullptr);
    if (it->memory) vkFreeMemory(it->device, it->memory, nullptr);

    if (it->cmdPool) vkDestroyCommandPool(it->device, it->cmdPool, nullptr);
    if (it->device) vkDestroyDevice(it->device, nullptr);
    if (it->instance) vkDestroyInstance(it->instance, nullptr);

    delete it;
}

extern "C" void* vulkaninterop_create(int64_t elem_count) {
    if (elem_count <= 0) {
        fprintf(stderr, "vulkaninterop: vulkaninterop_create called with elem_count <= 0\n");
        return nullptr;
    }

    Interop* it = new Interop();
    it->elemCount = elem_count;
    it->bufferBytes = (VkDeviceSize)elem_count * sizeof(float);

    ExtFns fns{};
    if (!createHeadlessInteropDevice(it, &fns)) { vulkaninterop_destroy(it); return nullptr; }

    // ---- Shared buffer: exportable memory, imported into CUDA ----
    {
        VkExternalMemoryBufferCreateInfo extBufInfo{};
        extBufInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        extBufInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.pNext = &extBufInfo;
        bci.size = it->bufferBytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK_GOTO(vkCreateBuffer(it->device, &bci, nullptr, &it->buffer), fail);

        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(it->device, it->buffer, &req);

        VkExportMemoryAllocateInfo exportInfo{};
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.pNext = &exportInfo;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = findMemoryType(it->physicalDevice, req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX) {
            fprintf(stderr, "vulkaninterop: no suitable device-local memory type for exportable buffer\n");
            goto fail;
        }
        VK_CHECK_GOTO(vkAllocateMemory(it->device, &mai, nullptr, &it->memory), fail);
        VK_CHECK_GOTO(vkBindBufferMemory(it->device, it->buffer, it->memory, 0), fail);

        int memFd = -1;
        VkMemoryGetFdInfoKHR getFdInfo{};
        getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        getFdInfo.memory = it->memory;
        getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VK_CHECK_GOTO(fns.getMemoryFd(it->device, &getFdInfo, &memFd), fail);

        // Ownership of memFd transfers to CUDA on a successful import --
        // must not close it ourselves either way below.
        cudaExternalMemoryHandleDesc memDesc{};
        memset(&memDesc, 0, sizeof(memDesc));
        memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memDesc.handle.fd = memFd;
        memDesc.size = req.size;
        CUDA_CHECK_GOTO(cudaImportExternalMemory(&it->cudaExtMem, &memDesc), fail);

        cudaExternalMemoryBufferDesc bufDesc{};
        memset(&bufDesc, 0, sizeof(bufDesc));
        bufDesc.offset = 0;
        bufDesc.size = it->bufferBytes;
        CUDA_CHECK_GOTO(cudaExternalMemoryGetMappedBuffer(&it->cudaPtr, it->cudaExtMem, &bufDesc), fail);
    }

    // ---- Two binary semaphores, each exported and imported into CUDA ----
    {
        VkExportSemaphoreCreateInfo exportSemInfo{};
        exportSemInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        exportSemInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkSemaphoreCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        sci.pNext = &exportSemInfo;

        VK_CHECK_GOTO(vkCreateSemaphore(it->device, &sci, nullptr, &it->semA), fail);
        VK_CHECK_GOTO(vkCreateSemaphore(it->device, &sci, nullptr, &it->semB), fail);

        for (int s = 0; s < 2; s++) {
            VkSemaphore vkSem = (s == 0) ? it->semA : it->semB;
            cudaExternalSemaphore_t* cudaSem = (s == 0) ? &it->cudaSemA : &it->cudaSemB;

            int semFd = -1;
            VkSemaphoreGetFdInfoKHR getFdInfo{};
            getFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
            getFdInfo.semaphore = vkSem;
            getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            VK_CHECK_GOTO(fns.getSemaphoreFd(it->device, &getFdInfo, &semFd), fail);

            cudaExternalSemaphoreHandleDesc semDesc{};
            memset(&semDesc, 0, sizeof(semDesc));
            semDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
            semDesc.handle.fd = semFd;
            CUDA_CHECK_GOTO(cudaImportExternalSemaphore(cudaSem, &semDesc), fail);
        }
    }

    // ---- Compute pipeline (interop_double.comp) ----
    {
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(interop_double_comp_spv);
        smci.pCode = interop_double_comp_spv;
        VK_CHECK_GOTO(vkCreateShaderModule(it->device, &smci, nullptr, &it->shaderModule), fail);

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 1;
        dslci.pBindings = &binding;
        VK_CHECK_GOTO(vkCreateDescriptorSetLayout(it->device, &dslci, nullptr, &it->dsLayout), fail);

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(uint32_t);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &it->dsLayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcRange;
        VK_CHECK_GOTO(vkCreatePipelineLayout(it->device, &plci, nullptr, &it->pipelineLayout), fail);

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = it->shaderModule;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = it->pipelineLayout;
        VK_CHECK_GOTO(vkCreateComputePipelines(it->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &it->pipeline), fail);

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &poolSize;
        VK_CHECK_GOTO(vkCreateDescriptorPool(it->device, &dpci, nullptr, &it->descPool), fail);

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = it->descPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &it->dsLayout;
        VK_CHECK_GOTO(vkAllocateDescriptorSets(it->device, &dsai, &it->descSet), fail);

        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = it->buffer;
        bufInfo.range = it->bufferBytes;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = it->descSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufInfo;
        vkUpdateDescriptorSets(it->device, 1, &write, 0, nullptr);

        // Pre-record the dispatch once -- semaphores are supplied at
        // vkQueueSubmit time (see vulkaninterop_round_trip), not baked into
        // the command buffer, so this recording is reused every round trip.
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = it->cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VK_CHECK_GOTO(vkAllocateCommandBuffers(it->device, &cbai, &it->cmd), fail);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        // Resubmitted every round trip with no reset/re-record in between.
        // The GPU-side semaphore chain guarantees each round trip's actual
        // EXECUTION is correctly ordered (the next round trip's CUDA-side
        // signal is enqueued after this one's CUDA-side wait, so on the
        // GPU timeline the previous dispatch is provably finished before
        // the next one starts) -- but that does NOT make host-side
        // resubmission of a still-possibly-pending command buffer valid on
        // its own: per the Vulkan spec, a command buffer not recorded with
        // SIMULTANEOUS_USE_BIT must not be re-submitted while it may still
        // be in the pending state, and the host has no way to know the GPU
        // has actually reached that point (cudaSignalExternalSemaphoresAsync/
        // vkQueueSubmit/cudaWaitExternalSemaphoresAsync are all async --
        // this function returns long before the GPU work completes). This
        // flag is what makes that host-side race legal; the GPU-side
        // ordering above is what makes it also produce correct results
        // (not just spec-legal but a data race).
        bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        vkBeginCommandBuffer(it->cmd, &bi);
        vkCmdBindPipeline(it->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, it->pipeline);
        vkCmdBindDescriptorSets(it->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, it->pipelineLayout, 0, 1, &it->descSet, 0, nullptr);
        uint32_t count32 = (uint32_t)it->elemCount;
        vkCmdPushConstants(it->cmd, it->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(count32), &count32);
        vkCmdDispatch(it->cmd, (count32 + 63) / 64, 1, 1);
        vkEndCommandBuffer(it->cmd);
    }

    return it;

fail:
    vulkaninterop_destroy(it);
    return nullptr;
}

extern "C" void* vulkaninterop_get_cuda_ptr(void* handle) {
    if (!handle) return nullptr;
    return ((Interop*)handle)->cudaPtr;
}

extern "C" int vulkaninterop_round_trip(void* handle, void* cuda_stream) {
    if (!handle) return 0;
    Interop* it = (Interop*)handle;
    cudaStream_t stream = (cudaStream_t)cuda_stream;

    // 1. CUDA signals semaphore A on `stream` -- fires only once every
    //    prior op enqueued on this stream (e.g. Mojo's "write" kernel) has
    //    actually completed on the GPU. Purely async: does not block this
    //    call.
    cudaExternalSemaphoreSignalParams sigParams;
    memset(&sigParams, 0, sizeof(sigParams));
    cudaError_t sc = cudaSignalExternalSemaphoresAsync(&it->cudaSemA, &sigParams, 1, stream);
    if (sc != cudaSuccess) {
        fprintf(stderr, "vulkaninterop: cudaSignalExternalSemaphoresAsync failed: %s\n", cudaGetErrorString(sc));
        return 0;
    }

    // 2. Submit the pre-recorded Vulkan dispatch: waits on A (GPU-side --
    //    the queue won't execute the compute shader until the semaphore is
    //    signaled), signals B when done. vkQueueSubmit itself only blocks
    //    for CPU-side command submission, not GPU completion.
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &it->semA;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &it->cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &it->semB;
    if (vkQueueSubmit(it->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkQueueSubmit failed\n");
        return 0;
    }

    // 3. CUDA waits on semaphore B, enqueued on `stream` -- any kernel Mojo
    //    enqueues on this stream AFTER this call (e.g. a "read results"
    //    kernel) will correctly wait for Vulkan's dispatch to finish,
    //    entirely on the GPU timeline. Async: does not block this call.
    cudaExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));
    cudaError_t wc = cudaWaitExternalSemaphoresAsync(&it->cudaSemB, &waitParams, 1, stream);
    if (wc != cudaSuccess) {
        fprintf(stderr, "vulkaninterop: cudaWaitExternalSemaphoresAsync failed: %s\n", cudaGetErrorString(wc));
        return 0;
    }

    return 1;
}

// ---------------------------------------------------------------------------
// Stage 2: real ray-query tracing through the interop mechanism.
// ---------------------------------------------------------------------------

extern "C" void vulkaninterop_rt_destroy_scene(void* handle) {
    if (!handle) return;
    InteropRtScene* sc = (InteropRtScene*)handle;

    // See vulkaninterop_destroy's matching comment -- required before
    // destroying device-owned objects, costs nothing in the render loop
    // since this only runs once at teardown.
    if (sc->device) vkDeviceWaitIdle(sc->device);

    if (sc->cudaSemA) cudaDestroyExternalSemaphore(sc->cudaSemA);
    if (sc->cudaSemB) cudaDestroyExternalSemaphore(sc->cudaSemB);
    if (sc->raysCudaExtMem) cudaDestroyExternalMemory(sc->raysCudaExtMem);
    if (sc->resultsCudaExtMem) cudaDestroyExternalMemory(sc->resultsCudaExtMem);

    if (sc->descPool) vkDestroyDescriptorPool(sc->device, sc->descPool, nullptr);
    if (sc->pipeline) vkDestroyPipeline(sc->device, sc->pipeline, nullptr);
    if (sc->pipelineLayout) vkDestroyPipelineLayout(sc->device, sc->pipelineLayout, nullptr);
    if (sc->dsLayout) vkDestroyDescriptorSetLayout(sc->device, sc->dsLayout, nullptr);
    if (sc->shaderModule) vkDestroyShaderModule(sc->device, sc->shaderModule, nullptr);

    if (sc->semA) vkDestroySemaphore(sc->device, sc->semA, nullptr);
    if (sc->semB) vkDestroySemaphore(sc->device, sc->semB, nullptr);
    if (sc->raysBuffer) vkDestroyBuffer(sc->device, sc->raysBuffer, nullptr);
    if (sc->raysMemory) vkFreeMemory(sc->device, sc->raysMemory, nullptr);
    if (sc->resultsBuffer) vkDestroyBuffer(sc->device, sc->resultsBuffer, nullptr);
    if (sc->resultsMemory) vkFreeMemory(sc->device, sc->resultsMemory, nullptr);

    if (sc->tlas) sc->rtFns.destroy(sc->device, sc->tlas, nullptr);
    destroyBuffer(sc->device, &sc->tlasBuf);
    destroyBuffer(sc->device, &sc->instanceBuf);

    for (size_t i = 0; i < sc->blas.size(); i++) {
        if (sc->blas[i]) sc->rtFns.destroy(sc->device, sc->blas[i], nullptr);
        destroyBuffer(sc->device, &sc->blasBufs[i]);
        destroyBuffer(sc->device, &sc->blasVertexBufs[i]);
        destroyBuffer(sc->device, &sc->blasIndexBufs[i]);
    }
    for (size_t t = 0; t < sc->templateBlas.size(); t++) {
        if (sc->templateBlas[t]) sc->rtFns.destroy(sc->device, sc->templateBlas[t], nullptr);
        destroyBuffer(sc->device, &sc->templateBlasBufs[t]);
    }

    if (sc->cmdPool) vkDestroyCommandPool(sc->device, sc->cmdPool, nullptr);
    if (sc->device) vkDestroyDevice(sc->device, nullptr);
    if (sc->instance) vkDestroyInstance(sc->instance, nullptr);

    delete sc;
}

extern "C" void* vulkaninterop_rt_create_scene(
    const VulkanInteropMesh* meshes,
    int64_t mesh_count,
    const int64_t* point_counts,
    const int64_t* vertex_index_counts,
    int64_t template_count,
    const int64_t* template_mesh_start,
    const int64_t* template_mesh_end,
    int64_t instance_count,
    const float* instance_obj_to_world,
    const int32_t* instance_template_idx,
    int64_t max_rays) {

    if (mesh_count <= 0 || max_rays <= 0) {
        fprintf(stderr, "vulkaninterop: vulkaninterop_rt_create_scene called with mesh_count/max_rays <= 0\n");
        return nullptr;
    }

    InteropRtScene* sc = new InteropRtScene();
    sc->maxRays = max_rays;
    sc->blas.resize((size_t)mesh_count, VK_NULL_HANDLE);
    sc->blasBufs.resize((size_t)mesh_count);
    sc->blasVertexBufs.resize((size_t)mesh_count);
    sc->blasIndexBufs.resize((size_t)mesh_count);
    sc->templateBlas.resize((size_t)template_count, VK_NULL_HANDLE);
    sc->templateBlasBufs.resize((size_t)template_count);

    if (!createHeadlessInteropRtDevice(sc)) {
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    // Which mesh indices belong to a template (excluded from the ordinary
    // one-BLAS-per-mesh-with-identity-transform loop below, folded instead
    // into their template's own multi-geometry BLAS).
    std::vector<bool> isTemplateMesh((size_t)mesh_count, false);
    for (int64_t t = 0; t < template_count; t++) {
        for (int64_t mi = template_mesh_start[t]; mi < template_mesh_end[t]; mi++) {
            isTemplateMesh[(size_t)mi] = true;
        }
    }

    // ---- Vertex/index buffers for EVERY mesh (ordinary and template alike
    //      -- template meshes' buffers get reused as geometries in their
    //      template's BLAS below, same upload logic either way) ----
    std::vector<VkAccelerationStructureGeometryKHR> meshGeoms((size_t)mesh_count);
    std::vector<uint32_t> meshTriCounts((size_t)mesh_count);
    for (int64_t i = 0; i < mesh_count; i++) {
        int64_t nVerts = point_counts[i];
        int64_t nIdx = vertex_index_counts[i];
        if (nVerts <= 0 || nIdx <= 0 || nIdx % 3 != 0) {
            fprintf(stderr, "vulkaninterop: mesh %lld has invalid vertex/index count (%lld/%lld)\n",
                    (long long)i, (long long)nVerts, (long long)nIdx);
            vulkaninterop_rt_destroy_scene(sc);
            return nullptr;
        }
        meshTriCounts[i] = (uint32_t)(nIdx / 3);

        VkDeviceSize vbBytes = (VkDeviceSize)nVerts * 4 * sizeof(float);
        if (!createBuffer(sc->device, sc->physicalDevice, vbBytes,
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           true, &sc->blasVertexBufs[i])) {
            vulkaninterop_rt_destroy_scene(sc);
            return nullptr;
        }
        if (!uploadToBuffer(sc->device, sc->blasVertexBufs[i], meshes[i].points, vbBytes)) {
            vulkaninterop_rt_destroy_scene(sc);
            return nullptr;
        }

        std::vector<uint32_t> indices32((size_t)nIdx);
        for (int64_t j = 0; j < nIdx; j++) {
            indices32[(size_t)j] = (uint32_t)meshes[i].vertexIndices[j];
        }
        VkDeviceSize ibBytes = (VkDeviceSize)nIdx * sizeof(uint32_t);
        if (!createBuffer(sc->device, sc->physicalDevice, ibBytes,
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           true, &sc->blasIndexBufs[i])) {
            vulkaninterop_rt_destroy_scene(sc);
            return nullptr;
        }
        if (!uploadToBuffer(sc->device, sc->blasIndexBufs[i], indices32.data(), ibBytes)) {
            vulkaninterop_rt_destroy_scene(sc);
            return nullptr;
        }

        VkAccelerationStructureGeometryKHR geom{};
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geom.geometry.triangles.vertexData.deviceAddress = sc->blasVertexBufs[i].address;
        geom.geometry.triangles.vertexStride = sizeof(float) * 4;
        geom.geometry.triangles.maxVertex = (uint32_t)(nVerts - 1);
        geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geom.geometry.triangles.indexData.deviceAddress = sc->blasIndexBufs[i].address;
        meshGeoms[i] = geom;
    }

    // ---- Ordinary meshes: one single-geometry BLAS each (identical to
    //      pre-instancing behavior) ----
    std::vector<VkDeviceAddress> blasAddresses((size_t)mesh_count, 0);
    for (int64_t i = 0; i < mesh_count; i++) {
        if (isTemplateMesh[(size_t)i]) continue;
        if (!buildAccelerationStructure(sc->device, sc->physicalDevice, sc->cmdPool,
                                         sc->queue, sc->rtFns,
                                         VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                         meshGeoms[i], meshTriCounts[i],
                                         &sc->blas[i], &sc->blasBufs[i], &blasAddresses[i])) {
            vulkaninterop_rt_destroy_scene(sc);
            return nullptr;
        }
    }

    // ---- Templates: one multi-geometry BLAS each, geometryIndex i-mstart
    //      within the BLAS corresponds to mesh (mstart+i) -- see
    //      intersect_batch.comp / vulkaninterop_unpack_results_kernel for
    //      the hit decode that relies on this exact indexing ----
    std::vector<VkDeviceAddress> templateBlasAddresses((size_t)template_count, 0);
    for (int64_t t = 0; t < template_count; t++) {
        int64_t mstart = template_mesh_start[t];
        int64_t mend   = template_mesh_end[t];
        std::vector<VkAccelerationStructureGeometryKHR> tGeoms(meshGeoms.begin() + mstart, meshGeoms.begin() + mend);
        std::vector<uint32_t> tCounts(meshTriCounts.begin() + mstart, meshTriCounts.begin() + mend);
        if (!buildAccelerationStructureMulti(sc->device, sc->physicalDevice, sc->cmdPool,
                                              sc->queue, sc->rtFns,
                                              tGeoms, tCounts,
                                              &sc->templateBlas[t], &sc->templateBlasBufs[t],
                                              &templateBlasAddresses[t])) {
            vulkaninterop_rt_destroy_scene(sc);
            return nullptr;
        }
    }

    // ---- TLAS instances: one per ordinary mesh (identity transform,
    //      instanceCustomIndex = mesh index, unchanged from pre-instancing)
    //      plus one per ObjectInstance placement (real transform,
    //      instanceCustomIndex = mesh_count + instance index -- decoded on
    //      the Mojo side, see vulkaninterop_unpack_results_kernel) ----
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve((size_t)(mesh_count + instance_count));
    for (int64_t i = 0; i < mesh_count; i++) {
        if (isTemplateMesh[(size_t)i]) continue;
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform.matrix[0][0] = 1; inst.transform.matrix[1][1] = 1; inst.transform.matrix[2][2] = 1;
        inst.instanceCustomIndex = (uint32_t)i;
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blasAddresses[i];
        instances.push_back(inst);
    }
    for (int64_t k = 0; k < instance_count; k++) {
        // gonzales's objToWorld is column-major (M[col*4+row], matches
        // transform.mojo/camera_to_world's own convention); Vulkan's
        // VkTransformMatrixKHR is the top 3 rows of a row-major 4x4, so
        // matrix[r][c] = M[c*4+r].
        const float* m = instance_obj_to_world + (size_t)k * 16;
        VkAccelerationStructureInstanceKHR inst{};
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 4; c++) {
                inst.transform.matrix[r][c] = m[c * 4 + r];
            }
        }
        inst.instanceCustomIndex = (uint32_t)(mesh_count + k);
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = templateBlasAddresses[(size_t)instance_template_idx[k]];
        instances.push_back(inst);
    }
    uint32_t instanceCount32 = (uint32_t)instances.size();
    VkDeviceSize instBytes = (VkDeviceSize)instances.size() * sizeof(VkAccelerationStructureInstanceKHR);
    if (!createBuffer(sc->device, sc->physicalDevice, instBytes,
                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       true, &sc->instanceBuf)) {
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }
    if (!uploadToBuffer(sc->device, sc->instanceBuf, instances.data(), instBytes)) {
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    VkAccelerationStructureGeometryKHR tlasGeom{};
    tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeom.geometry.instances.data.deviceAddress = sc->instanceBuf.address;

    VkDeviceAddress tlasAddress;
    if (!buildAccelerationStructure(sc->device, sc->physicalDevice, sc->cmdPool,
                                     sc->queue, sc->rtFns,
                                     VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                     tlasGeom, instanceCount32,
                                     &sc->tlas, &sc->tlasBuf, &tlasAddress)) {
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    // ---- The two CUDA-shared buffers (fixed for the scene's lifetime) ----
    sc->raysBytes = (VkDeviceSize)max_rays * 8 * sizeof(float);
    if (!createInteropBuffer(sc->device, sc->physicalDevice, sc->extFns, sc->raysBytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              &sc->raysBuffer, &sc->raysMemory, &sc->raysCudaExtMem, &sc->raysCudaPtr)) {
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }
    sc->resultsBytes = (VkDeviceSize)max_rays * 8 * sizeof(float); // Result is 32 bytes = 8 floats-equivalent
    if (!createInteropBuffer(sc->device, sc->physicalDevice, sc->extFns, sc->resultsBytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              &sc->resultsBuffer, &sc->resultsMemory, &sc->resultsCudaExtMem, &sc->resultsCudaPtr)) {
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    // ---- Two binary semaphores, CUDA-imported (same as stage 1) ----
    if (!createInteropSemaphore(sc->device, sc->extFns, &sc->semA, &sc->cudaSemA)) {
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }
    if (!createInteropSemaphore(sc->device, sc->extFns, &sc->semB, &sc->cudaSemB)) {
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    // ---- Compute pipeline: intersect_batch.comp, bound ONCE to the fixed
    //      rays/results buffers above (unlike vulkanrt.cpp's per-call
    //      rebinding -- these buffers never change for this scene) ----
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(intersect_batch_comp_spv);
    smci.pCode = intersect_batch_comp_spv;
    if (vkCreateShaderModule(sc->device, &smci, nullptr, &sc->shaderModule) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateShaderModule (rt) failed\n");
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(sc->device, &dslci, nullptr, &sc->dsLayout) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateDescriptorSetLayout (rt) failed\n");
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &sc->dsLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(sc->device, &plci, nullptr, &sc->pipelineLayout) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreatePipelineLayout (rt) failed\n");
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = sc->shaderModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    cpci.layout = sc->pipelineLayout;
    if (vkCreateComputePipelines(sc->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &sc->pipeline) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateComputePipelines (rt) failed\n");
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 2;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(sc->device, &dpci, nullptr, &sc->descPool) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkCreateDescriptorPool (rt) failed\n");
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = sc->descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &sc->dsLayout;
    if (vkAllocateDescriptorSets(sc->device, &dsai, &sc->descSet) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkAllocateDescriptorSets (rt) failed\n");
        vulkaninterop_rt_destroy_scene(sc);
        return nullptr;
    }

    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &sc->tlas;

    VkDescriptorBufferInfo raysInfo{};
    raysInfo.buffer = sc->raysBuffer;
    raysInfo.range = sc->raysBytes;
    VkDescriptorBufferInfo resultsInfo{};
    resultsInfo.buffer = sc->resultsBuffer;
    resultsInfo.range = sc->resultsBytes;

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &asWrite;
    writes[0].dstSet = sc->descSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = sc->descSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &raysInfo;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = sc->descSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &resultsInfo;
    vkUpdateDescriptorSets(sc->device, 3, writes, 0, nullptr);

    return sc;
}

extern "C" void* vulkaninterop_rt_get_rays_ptr(void* handle) {
    if (!handle) return nullptr;
    return ((InteropRtScene*)handle)->raysCudaPtr;
}

extern "C" void* vulkaninterop_rt_get_results_ptr(void* handle) {
    if (!handle) return nullptr;
    return ((InteropRtScene*)handle)->resultsCudaPtr;
}

extern "C" int vulkaninterop_rt_trace(void* handle, int32_t ray_count, void* cuda_stream) {
    if (!handle || ray_count <= 0) return 0;
    InteropRtScene* sc = (InteropRtScene*)handle;
    cudaStream_t stream = (cudaStream_t)cuda_stream;

    if ((int64_t)ray_count > sc->maxRays) {
        fprintf(stderr, "vulkaninterop: vulkaninterop_rt_trace ray_count %d exceeds max_rays %lld\n",
                ray_count, (long long)sc->maxRays);
        return 0;
    }
    uint32_t rayCountU32 = (uint32_t)ray_count;

    // Look up (or record, on first use) the command buffer for this exact
    // ray_count -- see the InteropRtScene::cmdCache field comment for why
    // this is cached-by-count rather than allocated/freed per call (a
    // pending-command-buffer-lifetime bug in an earlier version of this
    // function) or pre-recorded once (ray_count can legitimately vary,
    // e.g. a final partial wavefront batch).
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    for (auto& entry : sc->cmdCache) {
        if (entry.first == rayCountU32) { cmd = entry.second; break; }
    }
    if (cmd == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = sc->cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(sc->device, &cbai, &cmd) != VK_SUCCESS) {
            fprintf(stderr, "vulkaninterop: vkAllocateCommandBuffers failed in vulkaninterop_rt_trace\n");
            return 0;
        }

        // SIMULTANEOUS_USE_BIT: this command buffer will be resubmitted on
        // every future call with this same ray_count, with no CPU-side
        // fence/wait telling us a previous submission has finished first
        // (see vulkaninterop_round_trip's matching comment for the full
        // explanation of why this flag is required, not just a nice-to-have,
        // and why the GPU-side semaphore ordering is what makes it also
        // produce correct results rather than just being spec-legal).
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sc->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sc->pipelineLayout, 0, 1, &sc->descSet, 0, nullptr);
        vkCmdPushConstants(cmd, sc->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rayCountU32), &rayCountU32);
        vkCmdDispatch(cmd, (rayCountU32 + 63) / 64, 1, 1);
        vkEndCommandBuffer(cmd);

        sc->cmdCache.push_back({rayCountU32, cmd});
    }

    // Same async CUDA-signal -> Vulkan-submit(wait/signal) -> CUDA-wait
    // handoff as vulkaninterop_round_trip -- see that function's comments
    // for the full explanation. No vkFreeCommandBuffers here (unlike an
    // earlier version of this function) -- cmd stays alive in the cache
    // for reuse by future calls with the same ray_count, and is only
    // released when the whole scene is destroyed.
    cudaExternalSemaphoreSignalParams sigParams;
    memset(&sigParams, 0, sizeof(sigParams));
    cudaError_t sigErr = cudaSignalExternalSemaphoresAsync(&sc->cudaSemA, &sigParams, 1, stream);
    if (sigErr != cudaSuccess) {
        fprintf(stderr, "vulkaninterop: cudaSignalExternalSemaphoresAsync (rt) failed: %s\n", cudaGetErrorString(sigErr));
        return 0;
    }

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &sc->semA;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &sc->semB;
    if (vkQueueSubmit(sc->queue, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
        fprintf(stderr, "vulkaninterop: vkQueueSubmit (rt) failed\n");
        return 0;
    }

    cudaExternalSemaphoreWaitParams waitParams;
    memset(&waitParams, 0, sizeof(waitParams));
    cudaError_t waitErr = cudaWaitExternalSemaphoresAsync(&sc->cudaSemB, &waitParams, 1, stream);
    if (waitErr != cudaSuccess) {
        fprintf(stderr, "vulkaninterop: cudaWaitExternalSemaphoresAsync (rt) failed: %s\n", cudaGetErrorString(waitErr));
        return 0;
    }

    return 1;
}
