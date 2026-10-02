// Dumps the driver's own view of a compute pipeline via VK_KHR_pipeline_executable_properties:
// statistics (registers etc.) and internal representations (NVIDIA may expose assembly).
// Build: gcc -O1 pipeline_dump.c -I../../src/vulkanrt/generated -lvulkan -o /tmp/pipeline_dump
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "intersect_batch_comp_spv.h"
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1);} } while (0)
int main(int argc, char** argv) {
    const char* outdir = argc > 1 ? argv[1] : "/tmp/pipe_dump";
    char cmd[512]; snprintf(cmd, sizeof cmd, "mkdir -p %s", outdir); system(cmd);
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &n, &pd);
    const char* exts[] = {"VK_KHR_ray_query", "VK_KHR_acceleration_structure", "VK_KHR_deferred_host_operations",
                          "VK_KHR_pipeline_executable_properties", "VK_KHR_buffer_device_address"};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fpe = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    fpe.pipelineExecutableInfo = VK_TRUE;
    VkPhysicalDeviceRayQueryFeaturesKHR frq = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR}; frq.rayQuery = VK_TRUE; frq.pNext = &fpe;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR fas = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    fas.accelerationStructure = VK_TRUE; fas.pNext = &frq;
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE; f12.pNext = &fas;
    float prio = 1; VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = 0; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext = &f12;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 5; dci.ppEnabledExtensionNames = exts;
    VkDevice dev; CK(vkCreateDevice(pd, &dci, NULL, &dev));
    VkDescriptorSetLayoutBinding b[3] = {0};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dlci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dlci.bindingCount = 3; dlci.pBindings = b;
    VkDescriptorSetLayout dl; CK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dl; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = sizeof intersect_batch_comp_spv; smci.pCode = intersect_batch_comp_spv;
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &smci, NULL, &sm));
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm; cpci.stage.pName = "main"; cpci.layout = pl;
    VkPipelineCacheCreateInfo pcci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VkPipelineCache pcache; CK(vkCreatePipelineCache(dev, &pcci, NULL, &pcache));
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, pcache, 1, &cpci, NULL, &pipe));
    { size_t sz = 0; vkGetPipelineCacheData(dev, pcache, &sz, NULL); void* d = malloc(sz); vkGetPipelineCacheData(dev, pcache, &sz, d);
      char fn[512]; snprintf(fn, sizeof fn, "%s/pipeline_cache.bin", outdir); FILE* f = fopen(fn, "wb"); fwrite(d, 1, sz, f); fclose(f); printf("pipeline cache: %zu bytes\n", sz); }
    PFN_vkGetPipelineExecutablePropertiesKHR getProps = (void*)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR getStats = (void*)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableStatisticsKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR getIR = (void*)vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR}; pi.pipeline = pipe;
    uint32_t ne = 0; CK(getProps(dev, &pi, &ne, NULL));
    VkPipelineExecutablePropertiesKHR* props = calloc(ne, sizeof *props);
    for (uint32_t i = 0; i < ne; i++) props[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    CK(getProps(dev, &pi, &ne, props));
    printf("%u executables\n", ne);
    for (uint32_t e = 0; e < ne; e++) {
        printf("exe %u: %s | %s | subgroup %u\n", e, props[e].name, props[e].description, props[e].subgroupSize);
        VkPipelineExecutableInfoKHR xi = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR}; xi.pipeline = pipe; xi.executableIndex = e;
        uint32_t ns = 0; CK(getStats(dev, &xi, &ns, NULL));
        VkPipelineExecutableStatisticKHR* st = calloc(ns, sizeof *st);
        for (uint32_t i = 0; i < ns; i++) st[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
        CK(getStats(dev, &xi, &ns, st));
        for (uint32_t i = 0; i < ns; i++) printf("  stat %s = %llu (%s)\n", st[i].name, (unsigned long long)st[i].value.u64, st[i].description);
        uint32_t nr = 0; CK(getIR(dev, &xi, &nr, NULL));
        VkPipelineExecutableInternalRepresentationKHR* ir = calloc(nr, sizeof *ir);
        for (uint32_t i = 0; i < nr; i++) ir[i].sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
        CK(getIR(dev, &xi, &nr, ir));
        for (uint32_t i = 0; i < nr; i++) {
            ir[i].pData = malloc(ir[i].dataSize);
        }
        CK(getIR(dev, &xi, &nr, ir));
        for (uint32_t i = 0; i < nr; i++) {
            char fn[512]; snprintf(fn, sizeof fn, "%s/exe%u_ir%u_%s.%s", outdir, e, i, ir[i].name, ir[i].isText ? "txt" : "bin");
            for (char* c = fn + strlen(outdir) + 1; *c; c++) if (*c == ' ' || *c == '/') *c = '_';
            FILE* f = fopen(fn, "wb"); fwrite(ir[i].pData, 1, ir[i].dataSize, f); fclose(f);
            printf("  ir %u: %s (%s) %zu bytes text=%d -> %s\n", i, ir[i].name, ir[i].description, ir[i].dataSize, ir[i].isText, fn);
        }
    }
    return 0;
}
