// Task #162: Vulkan ray-tracing (VK_KHR_ray_query) bridge for gonzales's
// second GPU intersection backend.
//
// Unlike src/viewer/viewer.cpp (GLFW-tied, built for the interactive window
// and swapchain presentation), this bridge creates its OWN headless
// VkInstance/VkDevice with no windowing dependency at all -- gonzales's
// batch renders run headless (--gpu --vcm etc, no window), so the
// ray-tracing backend can't piggyback on the viewer's device. See
// project_vulkan_rt_backend memory for the full architecture rationale.
//
// Step 1 (vulkanrt_smoke_test): a fixed single-triangle scene + fixed ray,
// proving the Mojo -> C++ bridge -> Vulkan -> RT-core round trip works at
// all.
// Step 2 (vulkanrt_build_scene / vulkanrt_trace_ray / vulkanrt_destroy_scene):
// a real multi-mesh scene built from gonzales's own TriangleMesh_C data,
// traced with runtime (push-constant) rays.

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "vulkanrt.h"
#include "smoke_comp_spv.h"
#include "trace_ray_comp_spv.h"

// ---------------------------------------------------------------------------
// Error checking macro (mirrors viewer.cpp's VK_CHECK, adapted to this
// file's all-local-variables style -- jumps to `fail` instead of an early
// return, since cleanup here has many interdependent objects to tear down).
// ---------------------------------------------------------------------------

#define VK_CHECK_GOTO(call, label)                                       \
    do {                                                                 \
        VkResult _r = (call);                                            \
        if (_r != VK_SUCCESS) {                                          \
            fprintf(stderr, "vulkanrt: Vulkan error %d at %s:%d: %s\n", \
                    _r, __FILE__, __LINE__, #call);                      \
            goto label;                                                  \
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
// optional device address -- every buffer this bridge needs (vertex, index,
// instance, scratch, AS storage, result) fits this one shape.
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
        fprintf(stderr, "vulkanrt: vkCreateBuffer failed\n");
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
        fprintf(stderr, "vulkanrt: no suitable memory type for buffer\n");
        vkDestroyBuffer(device, out->buffer, nullptr);
        out->buffer = VK_NULL_HANDLE;
        return false;
    }
    if (vkAllocateMemory(device, &mai, nullptr, &out->memory) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkAllocateMemory failed\n");
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
        fprintf(stderr, "vulkanrt: vkMapMemory failed\n");
        return false;
    }
    memcpy(p, data, (size_t)bytes);
    vkUnmapMemory(device, buf.memory);
    return true;
}

// Manually-loaded KHR ray-tracing entry points -- these are extension
// functions, not part of the core 1.0-1.2 API, so they must be resolved via
// vkGetDeviceProcAddr rather than linked directly (portable across loader
// implementations; matches NVIDIA's own vk_raytracing_tutorial_KHR pattern).
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
        fprintf(stderr, "vulkanrt: failed to load one or more KHR acceleration structure entry points\n");
        return false;
    }
    return true;
}

// Creates a headless VkInstance + VkDevice (first physical device supporting
// VK_KHR_ray_query + VK_KHR_acceleration_structure + deferred host ops) with
// one compute-capable queue and its command pool. Shared by
// vulkanrt_smoke_test and vulkanrt_build_scene so the two stay consistent
// with each other's device requirements by construction, not by convention.
// On failure, whatever was created before the failing step is left non-null
// in its output param (matches every other function in this file's
// goto-cleanup style, and every caller here recovers via
// vulkanrt_destroy_scene-style cleanup that only touches non-null handles).
bool createHeadlessDevice(VkInstance* outInstance, VkPhysicalDevice* outPhysicalDevice,
                           VkDevice* outDevice, VkQueue* outQueue,
                           VkCommandPool* outCmdPool, RtFunctions* outFns) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "gonzales-vulkanrt";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, outInstance) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkCreateInstance failed\n");
        return false;
    }

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(*outInstance, &count, nullptr);
        if (count == 0) {
            fprintf(stderr, "vulkanrt: no Vulkan physical devices found\n");
            return false;
        }
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(*outInstance, &count, devices.data());
        for (auto pd : devices) {
            uint32_t extCount = 0;
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, nullptr);
            std::vector<VkExtensionProperties> exts(extCount);
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &extCount, exts.data());
            bool hasRQ = false, hasAS = false, hasDHO = false;
            for (auto& e : exts) {
                if (strcmp(e.extensionName, VK_KHR_RAY_QUERY_EXTENSION_NAME) == 0) hasRQ = true;
                if (strcmp(e.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0) hasAS = true;
                if (strcmp(e.extensionName, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) == 0) hasDHO = true;
            }
            if (hasRQ && hasAS && hasDHO) {
                physicalDevice = pd;
                break;
            }
        }
        if (!physicalDevice) {
            fprintf(stderr, "vulkanrt: no device supports VK_KHR_ray_query + VK_KHR_acceleration_structure\n");
            return false;
        }
    }
    *outPhysicalDevice = physicalDevice;

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qfs.data());
    uint32_t queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily = i; break; }
    }
    if (queueFamily == UINT32_MAX) {
        fprintf(stderr, "vulkanrt: no compute-capable queue family\n");
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

    const char* deviceExts[] = {
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
    };
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &vk12Features;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 3;
    dci.ppEnabledExtensionNames = deviceExts;
    if (vkCreateDevice(physicalDevice, &dci, nullptr, outDevice) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkCreateDevice failed\n");
        return false;
    }
    vkGetDeviceQueue(*outDevice, queueFamily, 0, outQueue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = queueFamily;
    if (vkCreateCommandPool(*outDevice, &cpci, nullptr, outCmdPool) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkCreateCommandPool failed\n");
        return false;
    }

    return loadRtFunctions(*outDevice, outFns);
}

// Builds either a BLAS (triangle geometry) or a TLAS (instance geometry),
// sharing the same query-size/allocate/scratch/build/submit sequence -- the
// two only differ in the VkAccelerationStructureGeometryKHR contents and the
// AS `type`.
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
        fprintf(stderr, "vulkanrt: vkCreateAccelerationStructureKHR failed\n");
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

} // namespace

extern "C" int vulkanrt_smoke_test(void) {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkAccelerationStructureKHR blas = VK_NULL_HANDLE, tlas = VK_NULL_HANDLE;
    Buffer vertexBuf{}, indexBuf{}, instanceBuf{}, blasBuf{}, tlasBuf{}, resultBuf{};
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    RtFunctions fns{};
    int result = 0;

    if (!createHeadlessDevice(&instance, &physicalDevice, &device, &queue, &cmdPool, &fns))
        goto cleanup;

    // ---- Triangle geometry: (0,0,0), (1,0,0), (0,1,0); ray at (0.25,0.25) ----
    {
        float vertices[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
        uint32_t indices[3] = {0, 1, 2};

        if (!createBuffer(device, physicalDevice, sizeof(vertices),
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           true, &vertexBuf)) goto cleanup;
        if (!createBuffer(device, physicalDevice, sizeof(indices),
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           true, &indexBuf)) goto cleanup;
        if (!uploadToBuffer(device, vertexBuf, vertices, sizeof(vertices))) goto cleanup;
        if (!uploadToBuffer(device, indexBuf, indices, sizeof(indices))) goto cleanup;

        VkAccelerationStructureGeometryKHR geom{};
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geom.geometry.triangles.vertexData.deviceAddress = vertexBuf.address;
        geom.geometry.triangles.vertexStride = sizeof(float) * 3;
        geom.geometry.triangles.maxVertex = 2;
        geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geom.geometry.triangles.indexData.deviceAddress = indexBuf.address;

        VkDeviceAddress blasAddress;
        if (!buildAccelerationStructure(device, physicalDevice, cmdPool, queue, fns,
                                         VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                         geom, 1, &blas, &blasBuf, &blasAddress)) goto cleanup;

        VkAccelerationStructureInstanceKHR inst{};
        inst.transform.matrix[0][0] = 1; inst.transform.matrix[1][1] = 1; inst.transform.matrix[2][2] = 1;
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blasAddress;

        if (!createBuffer(device, physicalDevice, sizeof(inst),
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           true, &instanceBuf)) goto cleanup;
        if (!uploadToBuffer(device, instanceBuf, &inst, sizeof(inst))) goto cleanup;

        VkAccelerationStructureGeometryKHR tlasGeom{};
        tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
        tlasGeom.geometry.instances.data.deviceAddress = instanceBuf.address;

        VkDeviceAddress tlasAddress;
        if (!buildAccelerationStructure(device, physicalDevice, cmdPool, queue, fns,
                                         VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                         tlasGeom, 1, &tlas, &tlasBuf, &tlasAddress)) goto cleanup;
    }

    // ---- Result buffer (host-visible so we can read it back directly) ----
    if (!createBuffer(device, physicalDevice, sizeof(float) + sizeof(uint32_t),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       false, &resultBuf)) goto cleanup;

    // ---- Compute pipeline ----
    {
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = sizeof(smoke_comp_spv);
        smci.pCode = smoke_comp_spv;
        VK_CHECK_GOTO(vkCreateShaderModule(device, &smci, nullptr, &shaderModule), cleanup);

        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 2;
        dslci.pBindings = bindings;
        VK_CHECK_GOTO(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsLayout), cleanup);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsLayout;
        VK_CHECK_GOTO(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout), cleanup);

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shaderModule;
        stage.pName = "main";

        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage = stage;
        cpci.layout = pipelineLayout;
        VK_CHECK_GOTO(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline), cleanup);

        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 1;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1;
        dpci.poolSizeCount = 2;
        dpci.pPoolSizes = poolSizes;
        VK_CHECK_GOTO(vkCreateDescriptorPool(device, &dpci, nullptr, &descPool), cleanup);

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = descPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsLayout;
        VkDescriptorSet descSet;
        VK_CHECK_GOTO(vkAllocateDescriptorSets(device, &dsai, &descSet), cleanup);

        VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
        asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        asWrite.accelerationStructureCount = 1;
        asWrite.pAccelerationStructures = &tlas;

        VkDescriptorBufferInfo bufInfo{};
        bufInfo.buffer = resultBuf.buffer;
        bufInfo.range = resultBuf.size;

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].pNext = &asWrite;
        writes[0].dstSet = descSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &bufInfo;
        vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        VK_CHECK_GOTO(vkAllocateCommandBuffers(device, &cbai, &cmd), cleanup);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bi);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdDispatch(cmd, 1, 1, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        vkFreeCommandBuffers(device, cmdPool, 1, &cmd);
    }

    // ---- Read back result ----
    {
        struct { float hitT; uint32_t hitFlag; } readback;
        void* p;
        vkMapMemory(device, resultBuf.memory, 0, sizeof(readback), 0, &p);
        memcpy(&readback, p, sizeof(readback));
        vkUnmapMemory(device, resultBuf.memory);

        if (readback.hitFlag == 1 && readback.hitT > 0.0f) {
            fprintf(stderr, "vulkanrt: smoke test PASSED (hitT=%f)\n", readback.hitT);
            result = 1;
        } else {
            fprintf(stderr, "vulkanrt: smoke test FAILED (hitFlag=%u hitT=%f, expected hit)\n",
                    readback.hitFlag, readback.hitT);
            result = 0;
        }
    }

cleanup:
    if (descPool) vkDestroyDescriptorPool(device, descPool, nullptr);
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (dsLayout) vkDestroyDescriptorSetLayout(device, dsLayout, nullptr);
    if (shaderModule) vkDestroyShaderModule(device, shaderModule, nullptr);
    if (tlas) fns.destroy(device, tlas, nullptr);
    if (blas) fns.destroy(device, blas, nullptr);
    destroyBuffer(device, &resultBuf);
    destroyBuffer(device, &tlasBuf);
    destroyBuffer(device, &blasBuf);
    destroyBuffer(device, &instanceBuf);
    destroyBuffer(device, &indexBuf);
    destroyBuffer(device, &vertexBuf);
    if (cmdPool) vkDestroyCommandPool(device, cmdPool, nullptr);
    if (device) vkDestroyDevice(device, nullptr);
    if (instance) vkDestroyInstance(instance, nullptr);
    return result;
}

// ---------------------------------------------------------------------------
// Step 2: real multi-mesh scene, runtime-ray tracing.
// ---------------------------------------------------------------------------

namespace {

// Owns every Vulkan object a built scene needs, including its own headless
// instance/device (mirrors vulkanrt_smoke_test's self-contained style --
// simplest correct thing; sharing one device across many built scenes is a
// later optimization, not needed for step 2 to be real and useful).
struct Scene {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    RtFunctions fns{};

    std::vector<Buffer> blasVertexBufs;
    std::vector<Buffer> blasIndexBufs;
    std::vector<VkAccelerationStructureKHR> blas;
    std::vector<Buffer> blasBufs;

    Buffer instanceBuf{};
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    Buffer tlasBuf{};

    Buffer resultBuf{};
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
};

} // namespace

extern "C" void vulkanrt_destroy_scene(void* sceneHandle) {
    if (!sceneHandle) return;
    Scene* scene = (Scene*)sceneHandle;

    if (scene->descPool) vkDestroyDescriptorPool(scene->device, scene->descPool, nullptr);
    if (scene->pipeline) vkDestroyPipeline(scene->device, scene->pipeline, nullptr);
    if (scene->pipelineLayout) vkDestroyPipelineLayout(scene->device, scene->pipelineLayout, nullptr);
    if (scene->dsLayout) vkDestroyDescriptorSetLayout(scene->device, scene->dsLayout, nullptr);
    if (scene->shaderModule) vkDestroyShaderModule(scene->device, scene->shaderModule, nullptr);
    destroyBuffer(scene->device, &scene->resultBuf);

    if (scene->tlas) scene->fns.destroy(scene->device, scene->tlas, nullptr);
    destroyBuffer(scene->device, &scene->tlasBuf);
    destroyBuffer(scene->device, &scene->instanceBuf);

    for (size_t i = 0; i < scene->blas.size(); i++) {
        if (scene->blas[i]) scene->fns.destroy(scene->device, scene->blas[i], nullptr);
        destroyBuffer(scene->device, &scene->blasBufs[i]);
        destroyBuffer(scene->device, &scene->blasVertexBufs[i]);
        destroyBuffer(scene->device, &scene->blasIndexBufs[i]);
    }

    if (scene->cmdPool) vkDestroyCommandPool(scene->device, scene->cmdPool, nullptr);
    if (scene->device) vkDestroyDevice(scene->device, nullptr);
    if (scene->instance) vkDestroyInstance(scene->instance, nullptr);

    delete scene;
}

extern "C" void* vulkanrt_build_scene(
    const VulkanRtMesh* meshes,
    int64_t mesh_count,
    const int64_t* point_counts,
    const int64_t* vertex_index_counts) {

    if (mesh_count <= 0) {
        fprintf(stderr, "vulkanrt: vulkanrt_build_scene called with mesh_count <= 0\n");
        return nullptr;
    }

    Scene* scene = new Scene();
    scene->blas.resize((size_t)mesh_count, VK_NULL_HANDLE);
    scene->blasBufs.resize((size_t)mesh_count);
    scene->blasVertexBufs.resize((size_t)mesh_count);
    scene->blasIndexBufs.resize((size_t)mesh_count);

    if (!createHeadlessDevice(&scene->instance, &scene->physicalDevice, &scene->device,
                               &scene->queue, &scene->cmdPool, &scene->fns)) {
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }

    std::vector<VkDeviceAddress> blasAddresses((size_t)mesh_count);

    for (int64_t i = 0; i < mesh_count; i++) {
        int64_t nVerts = point_counts[i];
        int64_t nIdx = vertex_index_counts[i];
        if (nVerts <= 0 || nIdx <= 0 || nIdx % 3 != 0) {
            fprintf(stderr, "vulkanrt: mesh %lld has invalid vertex/index count (%lld/%lld)\n",
                    (long long)i, (long long)nVerts, (long long)nIdx);
            vulkanrt_destroy_scene(scene);
            return nullptr;
        }
        uint32_t triCount = (uint32_t)(nIdx / 3);

        // Vertex buffer: gonzales's points are xyzw (stride 16B); Vulkan
        // reads a 3-float position out of that stride directly via
        // vertexStride, no repacking needed.
        VkDeviceSize vbBytes = (VkDeviceSize)nVerts * 4 * sizeof(float);
        if (!createBuffer(scene->device, scene->physicalDevice, vbBytes,
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           true, &scene->blasVertexBufs[i])) {
            vulkanrt_destroy_scene(scene);
            return nullptr;
        }
        if (!uploadToBuffer(scene->device, scene->blasVertexBufs[i], meshes[i].points, vbBytes)) {
            vulkanrt_destroy_scene(scene);
            return nullptr;
        }

        // Index buffer: gonzales stores vertexIndices as int64; Vulkan AS
        // builds only accept 8/16/32-bit index types, so narrow to uint32
        // here (real gonzales meshes never approach 2^32 vertices).
        std::vector<uint32_t> indices32((size_t)nIdx);
        for (int64_t j = 0; j < nIdx; j++) {
            indices32[(size_t)j] = (uint32_t)meshes[i].vertexIndices[j];
        }
        VkDeviceSize ibBytes = (VkDeviceSize)nIdx * sizeof(uint32_t);
        if (!createBuffer(scene->device, scene->physicalDevice, ibBytes,
                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           true, &scene->blasIndexBufs[i])) {
            vulkanrt_destroy_scene(scene);
            return nullptr;
        }
        if (!uploadToBuffer(scene->device, scene->blasIndexBufs[i], indices32.data(), ibBytes)) {
            vulkanrt_destroy_scene(scene);
            return nullptr;
        }

        VkAccelerationStructureGeometryKHR geom{};
        geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geom.geometry.triangles.vertexData.deviceAddress = scene->blasVertexBufs[i].address;
        geom.geometry.triangles.vertexStride = sizeof(float) * 4;
        geom.geometry.triangles.maxVertex = (uint32_t)(nVerts - 1);
        geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geom.geometry.triangles.indexData.deviceAddress = scene->blasIndexBufs[i].address;

        if (!buildAccelerationStructure(scene->device, scene->physicalDevice, scene->cmdPool,
                                         scene->queue, scene->fns,
                                         VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                         geom, triCount,
                                         &scene->blas[i], &scene->blasBufs[i], &blasAddresses[i])) {
            vulkanrt_destroy_scene(scene);
            return nullptr;
        }
    }

    // TLAS: one instance per mesh, identity transform (gonzales already
    // bakes each mesh's CTM into world-space point coordinates at parse
    // time -- see pbrt_parser.mojo's store_mesh). instanceCustomIndex =
    // mesh index, so trace_ray.comp can report which input mesh was hit.
    std::vector<VkAccelerationStructureInstanceKHR> instances((size_t)mesh_count);
    for (int64_t i = 0; i < mesh_count; i++) {
        VkAccelerationStructureInstanceKHR inst{};
        inst.transform.matrix[0][0] = 1; inst.transform.matrix[1][1] = 1; inst.transform.matrix[2][2] = 1;
        inst.instanceCustomIndex = (uint32_t)i;
        inst.mask = 0xFF;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blasAddresses[i];
        instances[i] = inst;
    }
    VkDeviceSize instBytes = (VkDeviceSize)mesh_count * sizeof(VkAccelerationStructureInstanceKHR);
    if (!createBuffer(scene->device, scene->physicalDevice, instBytes,
                       VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       true, &scene->instanceBuf)) {
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }
    if (!uploadToBuffer(scene->device, scene->instanceBuf, instances.data(), instBytes)) {
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }

    VkAccelerationStructureGeometryKHR tlasGeom{};
    tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeom.geometry.instances.data.deviceAddress = scene->instanceBuf.address;

    VkDeviceAddress tlasAddress;
    if (!buildAccelerationStructure(scene->device, scene->physicalDevice, scene->cmdPool,
                                     scene->queue, scene->fns,
                                     VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                     tlasGeom, (uint32_t)mesh_count,
                                     &scene->tlas, &scene->tlasBuf, &tlasAddress)) {
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }

    // Result buffer, reused across every vulkanrt_trace_ray call against
    // this scene.
    if (!createBuffer(scene->device, scene->physicalDevice,
                       sizeof(float) + 2 * sizeof(int32_t) + sizeof(uint32_t),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       false, &scene->resultBuf)) {
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }

    // Compute pipeline: trace_ray.comp takes the ray as push constants, so
    // this same pipeline/descriptor set is reused unchanged by every
    // vulkanrt_trace_ray call.
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = sizeof(trace_ray_comp_spv);
    smci.pCode = trace_ray_comp_spv;
    if (vkCreateShaderModule(scene->device, &smci, nullptr, &scene->shaderModule) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkCreateShaderModule failed\n");
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 2;
    dslci.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(scene->device, &dslci, nullptr, &scene->dsLayout) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkCreateDescriptorSetLayout failed\n");
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(float) * 8; // vec4 originAndTMin + vec4 dirAndTMax

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &scene->dsLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(scene->device, &plci, nullptr, &scene->pipelineLayout) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkCreatePipelineLayout failed\n");
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = scene->shaderModule;
    stage.pName = "main";

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = stage;
    cpci.layout = scene->pipelineLayout;
    if (vkCreateComputePipelines(scene->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &scene->pipeline) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkCreateComputePipelines failed\n");
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }

    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(scene->device, &dpci, nullptr, &scene->descPool) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkCreateDescriptorPool failed\n");
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = scene->descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &scene->dsLayout;
    if (vkAllocateDescriptorSets(scene->device, &dsai, &scene->descSet) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkAllocateDescriptorSets failed\n");
        vulkanrt_destroy_scene(scene);
        return nullptr;
    }

    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &scene->tlas;

    VkDescriptorBufferInfo bufInfo{};
    bufInfo.buffer = scene->resultBuf.buffer;
    bufInfo.range = scene->resultBuf.size;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &asWrite;
    writes[0].dstSet = scene->descSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = scene->descSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &bufInfo;
    vkUpdateDescriptorSets(scene->device, 2, writes, 0, nullptr);

    return scene;
}

extern "C" int vulkanrt_trace_ray(
    void* sceneHandle,
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float t_min, float t_max,
    float* out_t, int32_t* out_mesh, int32_t* out_triangle) {

    if (!sceneHandle) return 0;
    Scene* scene = (Scene*)sceneHandle;

    float pushConstants[8] = {ox, oy, oz, t_min, dx, dy, dz, t_max};

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = scene->cmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(scene->device, &cbai, &cmd) != VK_SUCCESS) {
        fprintf(stderr, "vulkanrt: vkAllocateCommandBuffers failed in vulkanrt_trace_ray\n");
        return 0;
    }

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scene->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scene->pipelineLayout, 0, 1, &scene->descSet, 0, nullptr);
    vkCmdPushConstants(cmd, scene->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
    vkCmdDispatch(cmd, 1, 1, 1);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(scene->queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(scene->queue);
    vkFreeCommandBuffers(scene->device, scene->cmdPool, 1, &cmd);

    struct { float hitT; int32_t hitMesh; int32_t hitTriangle; uint32_t hitFlag; } readback;
    void* p;
    vkMapMemory(scene->device, scene->resultBuf.memory, 0, sizeof(readback), 0, &p);
    memcpy(&readback, p, sizeof(readback));
    vkUnmapMemory(scene->device, scene->resultBuf.memory);

    if (readback.hitFlag == 1) {
        if (out_t) *out_t = readback.hitT;
        if (out_mesh) *out_mesh = readback.hitMesh;
        if (out_triangle) *out_triangle = readback.hitTriangle;
        return 1;
    }
    if (out_t) *out_t = -1.0f;
    if (out_mesh) *out_mesh = -1;
    if (out_triangle) *out_triangle = -1;
    return 0;
}
