// Task #162 step 1: headless Vulkan ray-query smoke test.
//
// Unlike src/viewer/viewer.cpp (GLFW-tied, built for the interactive window
// and swapchain presentation), this bridge creates its OWN headless
// VkInstance/VkDevice with no windowing dependency at all -- gonzales's
// batch renders run headless (--gpu --vcm etc, no window), so the
// ray-tracing backend can't piggyback on the viewer's device. See
// project_vulkan_rt_backend memory for the full architecture rationale.

#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "vulkanrt.h"
#include "smoke_comp_spv.h"

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
// optional device address -- every buffer this smoke test needs (vertex,
// index, instance, scratch, AS storage, result) fits this one shape.
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
    #define LOAD(name) \
        fns->name = nullptr
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
    #undef LOAD
    if (!fns->getBuildSizes || !fns->create || !fns->destroy ||
        !fns->cmdBuild || !fns->getDeviceAddress) {
        fprintf(stderr, "vulkanrt: failed to load one or more KHR acceleration structure entry points\n");
        return false;
    }
    return true;
}

// Builds either a BLAS (from a single hardcoded triangle) or a TLAS (from a
// single instance referencing that BLAS), sharing the same
// query-size/allocate/scratch/build/submit sequence -- the two only differ
// in the VkAccelerationStructureGeometryKHR contents and the AS `type`.
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

    // ---- Instance (headless -- no windowing extensions) ----
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "gonzales-vulkanrt-smoke";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VK_CHECK_GOTO(vkCreateInstance(&ici, nullptr, &instance), cleanup);

    // ---- Physical device: first one supporting the RT extensions ----
    {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        if (count == 0) {
            fprintf(stderr, "vulkanrt: no Vulkan physical devices found\n");
            goto cleanup;
        }
        std::vector<VkPhysicalDevice> devices(count);
        vkEnumeratePhysicalDevices(instance, &count, devices.data());
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
            goto cleanup;
        }
    }

    // ---- Logical device: one compute-capable queue + RT feature chain ----
    {
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
            goto cleanup;
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
        VK_CHECK_GOTO(vkCreateDevice(physicalDevice, &dci, nullptr, &device), cleanup);
        vkGetDeviceQueue(device, queueFamily, 0, &queue);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = queueFamily;
        VK_CHECK_GOTO(vkCreateCommandPool(device, &cpci, nullptr, &cmdPool), cleanup);
    }

    if (!loadRtFunctions(device, &fns)) goto cleanup;

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
        void* p;
        vkMapMemory(device, vertexBuf.memory, 0, sizeof(vertices), 0, &p);
        memcpy(p, vertices, sizeof(vertices));
        vkUnmapMemory(device, vertexBuf.memory);
        vkMapMemory(device, indexBuf.memory, 0, sizeof(indices), 0, &p);
        memcpy(p, indices, sizeof(indices));
        vkUnmapMemory(device, indexBuf.memory);

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
        vkMapMemory(device, instanceBuf.memory, 0, sizeof(inst), 0, &p);
        memcpy(p, &inst, sizeof(inst));
        vkUnmapMemory(device, instanceBuf.memory);

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
