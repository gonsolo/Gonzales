// Task #163 stage 1: CUDA/Vulkan GPU-side interop foundation. See
// vulkaninterop.h for the full rationale and mechanism description.

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "vulkaninterop.h"
#include "interop_double_comp_spv.h"

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

} // namespace

extern "C" void vulkaninterop_destroy(void* handle) {
    if (!handle) return;
    Interop* it = (Interop*)handle;

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
        // Reused across many vkQueueSubmit calls with no reset in between --
        // SIMULTANEOUS_USE would be needed for concurrent in-flight
        // resubmission, but vulkaninterop_round_trip only ever has one
        // submission of this command buffer in flight at a time (the next
        // round trip's CUDA-side signal is enqueued after this one's CUDA
        // wait, so they're ordered on the CUDA stream), so plain resubmit
        // without that flag is correct.
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
