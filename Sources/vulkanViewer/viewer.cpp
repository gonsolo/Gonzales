#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "viewer.h"
#include "fullscreen_vert_spv.h"
#include "fullscreen_frag_spv.h"

// ---------------------------------------------------------------------------
// Error checking macro
// ---------------------------------------------------------------------------

#define VK_CHECK(call)                                               \
    do {                                                             \
        VkResult _r = (call);                                        \
        if (_r != VK_SUCCESS) {                                      \
            fprintf(stderr, "Vulkan error %d at %s:%d: %s\n",       \
                    _r, __FILE__, __LINE__, #call);                  \
            return false;                                            \
        }                                                            \
    } while (0)

// ---------------------------------------------------------------------------
// Viewer struct (opaque handle)
// ---------------------------------------------------------------------------

struct Viewer {
    GLFWwindow* window = nullptr;
    int winWidth = 0, winHeight = 0;

    // Vulkan core
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsFamily = 0;

    // Swapchain
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent = {0, 0};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;

    // Render pass + framebuffers
    VkRenderPass renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers;

    // Pipeline
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    // Texture (renderer framebuffer)
    VkImage textureImage = VK_NULL_HANDLE;
    VkDeviceMemory textureMemory = VK_NULL_HANDLE;
    VkImageView textureImageView = VK_NULL_HANDLE;
    VkSampler textureSampler = VK_NULL_HANDLE;
    int texWidth = 0, texHeight = 0;
    bool textureReady = false;

    // Staging buffer
    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkDeviceSize stagingSize = 0;

    // Descriptors
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    // Commands
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    // Sync
    VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
    VkFence inFlightFence = VK_NULL_HANDLE;

    // Camera state
    CameraState camera = {};
    double lastMouseX = 0, lastMouseY = 0;
    bool mousePressed = false;
    bool firstMouse = true;
    float yaw = -90.0f, pitch = 0.0f;
    float moveSpeed = 0.5f;
    float sensitivity = 0.15f;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static uint32_t findMemoryType(VkPhysicalDevice pd, uint32_t filter,
                               VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(pd, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((filter & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    fprintf(stderr, "Failed to find suitable memory type\n");
    return UINT32_MAX;
}

static VkShaderModule createShaderModule(VkDevice device, const uint32_t* code,
                                         size_t size) {
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size;
    ci.pCode = code;
    VkShaderModule sm = VK_NULL_HANDLE;
    VkResult r = vkCreateShaderModule(device, &ci, nullptr, &sm);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "Failed to create shader module: %d\n", r);
    }
    return sm;
}

static bool transitionImageLayout(VkDevice device, VkCommandPool pool,
                                   VkQueue queue, VkImage image,
                                   VkImageLayout oldLayout,
                                   VkImageLayout newLayout) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(device, &ai, &cb));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStage, dstStage;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        // Initial transition for a new texture (content undefined)
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        // Re-upload: shader-read → transfer-dst
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else {
        fprintf(stderr, "Unsupported layout transition: %d -> %d\n",
                oldLayout, newLayout);
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = 0;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    }

    vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1,
                         &barrier);

    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, pool, 1, &cb);
    return true;
}

// ---------------------------------------------------------------------------
// GLFW callbacks
// ---------------------------------------------------------------------------

static void keyCallback(GLFWwindow* window, int key, int /*scancode*/,
                        int action, int /*mods*/) {
    auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        float speed = v->moveSpeed;
        float dx = v->camera.dirX, dy = v->camera.dirY, dz = v->camera.dirZ;
        float ux = v->camera.upX, uy = v->camera.upY, uz = v->camera.upZ;
        float rx = dy * uz - dz * uy;
        float ry = dz * ux - dx * uz;
        float rz = dx * uy - dy * ux;

        bool moved = true;
        switch (key) {
            case GLFW_KEY_W:
                v->camera.posX += dx * speed;
                v->camera.posY += dy * speed;
                v->camera.posZ += dz * speed;
                break;
            case GLFW_KEY_S:
                v->camera.posX -= dx * speed;
                v->camera.posY -= dy * speed;
                v->camera.posZ -= dz * speed;
                break;
            case GLFW_KEY_A:
                v->camera.posX -= rx * speed;
                v->camera.posY -= ry * speed;
                v->camera.posZ -= rz * speed;
                break;
            case GLFW_KEY_D:
                v->camera.posX += rx * speed;
                v->camera.posY += ry * speed;
                v->camera.posZ += rz * speed;
                break;
            case GLFW_KEY_Q:
                v->camera.posX += ux * speed;
                v->camera.posY += uy * speed;
                v->camera.posZ += uz * speed;
                break;
            case GLFW_KEY_E:
                v->camera.posX -= ux * speed;
                v->camera.posY -= uy * speed;
                v->camera.posZ -= uz * speed;
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                moved = false;
                break;
            default:
                moved = false;
                break;
        }
        if (moved) v->camera.cameraChanged = 1;
    }
}

static void mouseButtonCallback(GLFWwindow* window, int button, int action,
                                 int /*mods*/) {
    auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        v->mousePressed = (action == GLFW_PRESS);
        if (v->mousePressed) {
            glfwGetCursorPos(window, &v->lastMouseX, &v->lastMouseY);
            v->firstMouse = true;
        }
    }
}

static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
    auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
    if (!v->mousePressed) return;

    if (v->firstMouse) {
        v->lastMouseX = xpos;
        v->lastMouseY = ypos;
        v->firstMouse = false;
        return;
    }

    float xoffset = float(xpos - v->lastMouseX) * v->sensitivity;
    float yoffset = float(v->lastMouseY - ypos) * v->sensitivity;
    v->lastMouseX = xpos;
    v->lastMouseY = ypos;

    v->yaw += xoffset;
    v->pitch += yoffset;

    if (v->pitch > 89.0f) v->pitch = 89.0f;
    if (v->pitch < -89.0f) v->pitch = -89.0f;

    float yawRad = v->yaw * 3.14159265f / 180.0f;
    float pitchRad = v->pitch * 3.14159265f / 180.0f;

    v->camera.dirX = cosf(pitchRad) * sinf(yawRad);
    v->camera.dirY = sinf(pitchRad);
    v->camera.dirZ = -cosf(pitchRad) * cosf(yawRad);

    float len = sqrtf(v->camera.dirX * v->camera.dirX +
                       v->camera.dirY * v->camera.dirY +
                       v->camera.dirZ * v->camera.dirZ);
    v->camera.dirX /= len;
    v->camera.dirY /= len;
    v->camera.dirZ /= len;

    v->camera.cameraChanged = 1;
}

static void scrollCallback(GLFWwindow* window, double /*xoffset*/,
                            double yoffset) {
    auto* v = static_cast<Viewer*>(glfwGetWindowUserPointer(window));
    v->moveSpeed *= (yoffset > 0) ? 1.2f : 0.8f;
    if (v->moveSpeed < 0.001f) v->moveSpeed = 0.001f;
    if (v->moveSpeed > 100.0f) v->moveSpeed = 100.0f;
}

// ---------------------------------------------------------------------------
// Vulkan setup helpers — each returns false on failure
// ---------------------------------------------------------------------------

static bool createInstance(Viewer* v) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "Gonzales Viewer";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "Gonzales";
    app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion = VK_API_VERSION_1_0;

    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    if (!glfwExts) {
        fprintf(stderr, "GLFW: Vulkan not supported on this platform\n");
        return false;
    }

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = glfwExtCount;
    ci.ppEnabledExtensionNames = glfwExts;
    ci.enabledLayerCount = 0;

    VK_CHECK(vkCreateInstance(&ci, nullptr, &v->instance));
    printf("Viewer: Vulkan instance created\n");
    return true;
}

static bool pickPhysicalDevice(Viewer* v) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(v->instance, &count, nullptr);
    if (count == 0) {
        fprintf(stderr, "No Vulkan physical devices found\n");
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(v->instance, &count, devices.data());

    for (auto& pd : devices) {
        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());

        for (uint32_t i = 0; i < qfCount; i++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, v->surface, &present);
            if ((qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
                v->physicalDevice = pd;
                v->graphicsFamily = i;

                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(pd, &props);
                printf("Viewer: Using GPU: %s\n", props.deviceName);
                return true;
            }
        }
    }
    fprintf(stderr, "No suitable Vulkan device found (need graphics + present)\n");
    return false;
}

static bool createDevice(Viewer* v) {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = v->graphicsFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const char* ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = &ext;

    VK_CHECK(vkCreateDevice(v->physicalDevice, &dci, nullptr, &v->device));
    vkGetDeviceQueue(v->device, v->graphicsFamily, 0, &v->graphicsQueue);
    printf("Viewer: Logical device created\n");
    return true;
}

static bool createSwapchain(Viewer* v) {
    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        v->physicalDevice, v->surface, &caps));

    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(v->physicalDevice, v->surface,
                                          &fmtCount, nullptr);
    if (fmtCount == 0) {
        fprintf(stderr, "No surface formats available\n");
        return false;
    }
    std::vector<VkSurfaceFormatKHR> fmts(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(v->physicalDevice, v->surface,
                                          &fmtCount, fmts.data());

    VkSurfaceFormatKHR chosen = fmts[0];
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    v->swapchainFormat = chosen.format;

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == UINT32_MAX) {
        int w, h;
        glfwGetFramebufferSize(v->window, &w, &h);
        extent.width = static_cast<uint32_t>(w);
        extent.height = static_cast<uint32_t>(h);
    }

    // Guard against zero-size swapchain (minimized window)
    if (extent.width == 0 || extent.height == 0) {
        fprintf(stderr, "Swapchain extent is zero (%ux%u) — is the window visible?\n",
                extent.width, extent.height);
        return false;
    }
    v->swapchainExtent = extent;

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
        imageCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sci{};
    sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    sci.surface = v->surface;
    sci.minImageCount = imageCount;
    sci.imageFormat = chosen.format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sci.clipped = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(v->device, &sci, nullptr, &v->swapchain));

    vkGetSwapchainImagesKHR(v->device, v->swapchain, &imageCount, nullptr);
    v->swapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(v->device, v->swapchain, &imageCount,
                             v->swapchainImages.data());

    v->swapchainImageViews.resize(imageCount);
    for (uint32_t i = 0; i < imageCount; i++) {
        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = v->swapchainImages[i];
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = v->swapchainFormat;
        ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        ivci.subresourceRange.baseMipLevel = 0;
        ivci.subresourceRange.levelCount = 1;
        ivci.subresourceRange.baseArrayLayer = 0;
        ivci.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(v->device, &ivci, nullptr,
                                    &v->swapchainImageViews[i]));
    }
    printf("Viewer: Swapchain created (%ux%u, %u images)\n",
           extent.width, extent.height, imageCount);
    return true;
}

static bool createRenderPass(Viewer* v) {
    VkAttachmentDescription colorAtt{};
    colorAtt.format = v->swapchainFormat;
    colorAtt.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{};
    ref.attachment = 0;
    ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass = 0;
    dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments = &colorAtt;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies = &dep;

    VK_CHECK(vkCreateRenderPass(v->device, &rpci, nullptr, &v->renderPass));
    return true;
}

static bool createFramebuffers(Viewer* v) {
    v->framebuffers.resize(v->swapchainImageViews.size());
    for (size_t i = 0; i < v->swapchainImageViews.size(); i++) {
        VkFramebufferCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass = v->renderPass;
        fci.attachmentCount = 1;
        fci.pAttachments = &v->swapchainImageViews[i];
        fci.width = v->swapchainExtent.width;
        fci.height = v->swapchainExtent.height;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(v->device, &fci, nullptr,
                                      &v->framebuffers[i]));
    }
    return true;
}

static bool createDescriptorSetLayout(Viewer* v) {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 1;
    ci.pBindings = &binding;
    VK_CHECK(vkCreateDescriptorSetLayout(v->device, &ci, nullptr,
                                          &v->descriptorSetLayout));
    return true;
}

static bool createPipeline(Viewer* v) {
    VkShaderModule vertMod =
        createShaderModule(v->device, fullscreen_vert_spv, sizeof(fullscreen_vert_spv));
    VkShaderModule fragMod =
        createShaderModule(v->device, fullscreen_frag_spv, sizeof(fullscreen_frag_spv));

    if (vertMod == VK_NULL_HANDLE || fragMod == VK_NULL_HANDLE) {
        fprintf(stderr, "Failed to create shader modules\n");
        if (vertMod) vkDestroyShaderModule(v->device, vertMod, nullptr);
        if (fragMod) vkDestroyShaderModule(v->device, fragMod, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertInput{};
    vertInput.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAsm{};
    inputAsm.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAsm.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.x = 0;
    viewport.y = 0;
    viewport.width = (float)v->swapchainExtent.width;
    viewport.height = (float)v->swapchainExtent.height;
    viewport.minDepth = 0;
    viewport.maxDepth = 1;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = v->swapchainExtent;

    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1;
    vpState.pViewports = &viewport;
    vpState.scissorCount = 1;
    vpState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.lineWidth = 1.0f;
    rast.cullMode = VK_CULL_MODE_NONE;
    rast.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    cba.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &v->descriptorSetLayout;
    VK_CHECK(vkCreatePipelineLayout(v->device, &plci, nullptr,
                                     &v->pipelineLayout));

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vertInput;
    pci.pInputAssemblyState = &inputAsm;
    pci.pViewportState = &vpState;
    pci.pRasterizationState = &rast;
    pci.pMultisampleState = &ms;
    pci.pColorBlendState = &cb;
    pci.layout = v->pipelineLayout;
    pci.renderPass = v->renderPass;
    pci.subpass = 0;

    VK_CHECK(vkCreateGraphicsPipelines(v->device, VK_NULL_HANDLE, 1, &pci,
                                        nullptr, &v->pipeline));

    vkDestroyShaderModule(v->device, vertMod, nullptr);
    vkDestroyShaderModule(v->device, fragMod, nullptr);
    return true;
}

static bool createTexture(Viewer* v, int width, int height) {
    v->texWidth = width;
    v->texHeight = height;

    // RGBA32F texture for HDR framebuffer
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ici.extent = {(uint32_t)width, (uint32_t)height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(v->device, &ici, nullptr, &v->textureImage));

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(v->device, v->textureImage, &memReq);

    uint32_t memType = findMemoryType(v->physicalDevice, memReq.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memType == UINT32_MAX) return false;

    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = memReq.size;
    mai.memoryTypeIndex = memType;

    VK_CHECK(vkAllocateMemory(v->device, &mai, nullptr, &v->textureMemory));
    VK_CHECK(vkBindImageMemory(v->device, v->textureImage, v->textureMemory, 0));

    // Image view
    VkImageViewCreateInfo ivci{};
    ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ivci.image = v->textureImage;
    ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    ivci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    ivci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    ivci.subresourceRange.baseMipLevel = 0;
    ivci.subresourceRange.levelCount = 1;
    ivci.subresourceRange.baseArrayLayer = 0;
    ivci.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(v->device, &ivci, nullptr, &v->textureImageView));

    // Sampler
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(v->device, &sci, nullptr, &v->textureSampler));

    // Staging buffer (RGBA32F = 16 bytes per pixel)
    v->stagingSize = (VkDeviceSize)width * height * 16;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = v->stagingSize;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(v->device, &bci, nullptr, &v->stagingBuffer));

    VkMemoryRequirements bufReq;
    vkGetBufferMemoryRequirements(v->device, v->stagingBuffer, &bufReq);

    uint32_t bufMemType = findMemoryType(
        v->physicalDevice, bufReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (bufMemType == UINT32_MAX) return false;

    VkMemoryAllocateInfo bmai{};
    bmai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    bmai.allocationSize = bufReq.size;
    bmai.memoryTypeIndex = bufMemType;
    VK_CHECK(vkAllocateMemory(v->device, &bmai, nullptr, &v->stagingMemory));
    VK_CHECK(vkBindBufferMemory(v->device, v->stagingBuffer, v->stagingMemory, 0));

    // Transition texture to shader-read-only (initially black)
    if (!transitionImageLayout(v->device, v->commandPool, v->graphicsQueue,
                               v->textureImage, VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)) {
        return false;
    }

    v->textureReady = true;
    printf("Viewer: Texture created (%dx%d, RGBA32F)\n", width, height);
    return true;
}

static bool createDescriptors(Viewer* v) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &poolSize;
    dpci.maxSets = 1;
    VK_CHECK(vkCreateDescriptorPool(v->device, &dpci, nullptr,
                                     &v->descriptorPool));

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = v->descriptorPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &v->descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(v->device, &dsai, &v->descriptorSet));

    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    imgInfo.imageView = v->textureImageView;
    imgInfo.sampler = v->textureSampler;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = v->descriptorSet;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imgInfo;
    vkUpdateDescriptorSets(v->device, 1, &write, 0, nullptr);
    return true;
}

static bool createCommandPool(Viewer* v) {
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = v->graphicsFamily;
    VK_CHECK(vkCreateCommandPool(v->device, &cpci, nullptr, &v->commandPool));

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = v->commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(v->device, &cbai, &v->commandBuffer));
    return true;
}

static bool createSyncObjects(Viewer* v) {
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    VK_CHECK(vkCreateSemaphore(v->device, &sci, nullptr,
                                &v->imageAvailableSemaphore));
    VK_CHECK(vkCreateSemaphore(v->device, &sci, nullptr,
                                &v->renderFinishedSemaphore));
    VK_CHECK(vkCreateFence(v->device, &fci, nullptr, &v->inFlightFence));
    return true;
}

static void drawFrame(Viewer* v) {
    vkWaitForFences(v->device, 1, &v->inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(v->device, 1, &v->inFlightFence);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(v->device, v->swapchain,
                                             UINT64_MAX,
                                             v->imageAvailableSemaphore,
                                             VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) return;

    vkResetCommandBuffer(v->commandBuffer, 0);

    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(v->commandBuffer, &cbbi);

    VkClearValue clear{};
    clear.color = {{0.05f, 0.05f, 0.05f, 1.0f}};

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = v->renderPass;
    rpbi.framebuffer = v->framebuffers[imageIndex];
    rpbi.renderArea.offset = {0, 0};
    rpbi.renderArea.extent = v->swapchainExtent;
    rpbi.clearValueCount = 1;
    rpbi.pClearValues = &clear;

    vkCmdBeginRenderPass(v->commandBuffer, &rpbi,
                          VK_SUBPASS_CONTENTS_INLINE);

    if (v->textureReady) {
        vkCmdBindPipeline(v->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          v->pipeline);
        vkCmdBindDescriptorSets(v->commandBuffer,
                                 VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 v->pipelineLayout, 0, 1,
                                 &v->descriptorSet, 0, nullptr);
        vkCmdDraw(v->commandBuffer, 3, 1, 0, 0);
    }

    vkCmdEndRenderPass(v->commandBuffer);
    vkEndCommandBuffer(v->commandBuffer);

    VkPipelineStageFlags waitStage =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &v->imageAvailableSemaphore;
    si.pWaitDstStageMask = &waitStage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &v->commandBuffer;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &v->renderFinishedSemaphore;
    vkQueueSubmit(v->graphicsQueue, 1, &si, v->inFlightFence);

    VkPresentInfoKHR pi{};
    pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &v->renderFinishedSemaphore;
    pi.swapchainCount = 1;
    pi.pSwapchains = &v->swapchain;
    pi.pImageIndices = &imageIndex;
    vkQueuePresentKHR(v->graphicsQueue, &pi);
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

extern "C" {

ViewerHandle viewer_create(int width, int height, const char* title) {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return nullptr;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    auto* v = new Viewer{};
    v->winWidth = width;
    v->winHeight = height;

    // Default camera: look along -Z
    v->camera.posX = 0;  v->camera.posY = 0;  v->camera.posZ = 0;
    v->camera.dirX = 0;  v->camera.dirY = 0;  v->camera.dirZ = -1;
    v->camera.upX = 0;   v->camera.upY = 1;   v->camera.upZ = 0;
    v->camera.cameraChanged = 0;

    v->window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!v->window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        delete v;
        glfwTerminate();
        return nullptr;
    }

    glfwSetWindowUserPointer(v->window, v);
    glfwSetKeyCallback(v->window, keyCallback);
    glfwSetMouseButtonCallback(v->window, mouseButtonCallback);
    glfwSetCursorPosCallback(v->window, cursorPosCallback);
    glfwSetScrollCallback(v->window, scrollCallback);

    // Vulkan setup — each step validates and returns false on failure
    bool ok = true;
    ok = ok && createInstance(v);
    if (ok) {
        VkResult sr = glfwCreateWindowSurface(v->instance, v->window, nullptr,
                                               &v->surface);
        if (sr != VK_SUCCESS) {
            fprintf(stderr, "Failed to create window surface: %d\n", sr);
            ok = false;
        }
    }
    ok = ok && pickPhysicalDevice(v);
    ok = ok && createDevice(v);
    ok = ok && createCommandPool(v);
    ok = ok && createSwapchain(v);
    ok = ok && createRenderPass(v);
    ok = ok && createFramebuffers(v);
    ok = ok && createDescriptorSetLayout(v);
    ok = ok && createPipeline(v);
    ok = ok && createTexture(v, width, height);
    ok = ok && createDescriptors(v);
    ok = ok && createSyncObjects(v);

    if (!ok) {
        fprintf(stderr, "Viewer: Vulkan initialization failed\n");
        // Partial cleanup
        if (v->device) vkDeviceWaitIdle(v->device);
        glfwDestroyWindow(v->window);
        glfwTerminate();
        delete v;
        return nullptr;
    }

    printf("Viewer: %dx%d window ready\n", width, height);
    return v;
}

void viewer_update_framebuffer(ViewerHandle viewer, const float* pixels,
                                int width, int height) {
    auto* v = static_cast<Viewer*>(viewer);
    if (!v || !pixels) return;
    if (v->device == VK_NULL_HANDLE) return;

    // Recreate texture if size changed
    if (width != v->texWidth || height != v->texHeight) {
        vkDeviceWaitIdle(v->device);
        vkDestroyImageView(v->device, v->textureImageView, nullptr);
        vkDestroySampler(v->device, v->textureSampler, nullptr);
        vkDestroyImage(v->device, v->textureImage, nullptr);
        vkFreeMemory(v->device, v->textureMemory, nullptr);
        vkDestroyBuffer(v->device, v->stagingBuffer, nullptr);
        vkFreeMemory(v->device, v->stagingMemory, nullptr);
        vkDestroyDescriptorPool(v->device, v->descriptorPool, nullptr);
        if (!createTexture(v, width, height) || !createDescriptors(v)) {
            fprintf(stderr, "Failed to recreate texture\n");
            return;
        }
    }

    // Wait for any in-flight rendering to complete before touching staging buffer
    vkDeviceWaitIdle(v->device);

    // Copy RGB float data → RGBA32F staging buffer
    void* data;
    VkResult mapResult = vkMapMemory(v->device, v->stagingMemory, 0,
                                      v->stagingSize, 0, &data);
    if (mapResult != VK_SUCCESS) {
        fprintf(stderr, "vkMapMemory failed: %d\n", mapResult);
        return;
    }
    float* dst = static_cast<float*>(data);
    int numPixels = width * height;
    for (int i = 0; i < numPixels; i++) {
        dst[i * 4 + 0] = pixels[i * 3 + 0];
        dst[i * 4 + 1] = pixels[i * 3 + 1];
        dst[i * 4 + 2] = pixels[i * 3 + 2];
        dst[i * 4 + 3] = 1.0f;
    }
    vkUnmapMemory(v->device, v->stagingMemory);

    // Transition → transfer dst
    transitionImageLayout(v->device, v->commandPool, v->graphicsQueue,
                          v->textureImage,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Copy staging buffer → texture image
    {
        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = v->commandPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb;
        vkAllocateCommandBuffers(v->device, &ai, &cb);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {(uint32_t)width, (uint32_t)height, 1};

        vkCmdCopyBufferToImage(cb, v->stagingBuffer, v->textureImage,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                &region);

        vkEndCommandBuffer(cb);

        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        vkQueueSubmit(v->graphicsQueue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(v->graphicsQueue);
        vkFreeCommandBuffers(v->device, v->commandPool, 1, &cb);
    }

    // Transition → shader read
    transitionImageLayout(v->device, v->commandPool, v->graphicsQueue,
                          v->textureImage,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Draw frame
    drawFrame(v);
}

int viewer_should_close(ViewerHandle viewer) {
    auto* v = static_cast<Viewer*>(viewer);
    return v ? glfwWindowShouldClose(v->window) : 1;
}

void viewer_poll_events(ViewerHandle viewer) {
    (void)viewer;
    glfwPollEvents();
}

CameraState viewer_get_camera_state(ViewerHandle viewer) {
    auto* v = static_cast<Viewer*>(viewer);
    CameraState state = v->camera;
    v->camera.cameraChanged = 0;
    return state;
}

void viewer_set_camera_state(ViewerHandle viewer, CameraState state) {
    auto* v = static_cast<Viewer*>(viewer);
    if (!v) return;
    v->camera = state;

    // Based on the set camera, we should also compute pitch and yaw to match.
    // dirX, dirY, dirZ should already be normalized.
    float pitchRad = asinf(v->camera.dirY);
    float yawRad = atan2f(v->camera.dirX, -v->camera.dirZ);
    v->pitch = pitchRad * 180.0f / 3.14159265f;
    v->yaw = yawRad * 180.0f / 3.14159265f;
}

void viewer_destroy(ViewerHandle viewer) {
    auto* v = static_cast<Viewer*>(viewer);
    if (!v) return;

    if (v->device) {
        vkDeviceWaitIdle(v->device);

        vkDestroySemaphore(v->device, v->imageAvailableSemaphore, nullptr);
        vkDestroySemaphore(v->device, v->renderFinishedSemaphore, nullptr);
        vkDestroyFence(v->device, v->inFlightFence, nullptr);

        vkDestroyDescriptorPool(v->device, v->descriptorPool, nullptr);
        vkDestroyDescriptorSetLayout(v->device, v->descriptorSetLayout, nullptr);

        vkDestroySampler(v->device, v->textureSampler, nullptr);
        vkDestroyImageView(v->device, v->textureImageView, nullptr);
        vkDestroyImage(v->device, v->textureImage, nullptr);
        vkFreeMemory(v->device, v->textureMemory, nullptr);
        vkDestroyBuffer(v->device, v->stagingBuffer, nullptr);
        vkFreeMemory(v->device, v->stagingMemory, nullptr);

        vkDestroyPipeline(v->device, v->pipeline, nullptr);
        vkDestroyPipelineLayout(v->device, v->pipelineLayout, nullptr);
        vkDestroyRenderPass(v->device, v->renderPass, nullptr);

        for (auto fb : v->framebuffers)
            vkDestroyFramebuffer(v->device, fb, nullptr);
        for (auto iv : v->swapchainImageViews)
            vkDestroyImageView(v->device, iv, nullptr);

        vkDestroySwapchainKHR(v->device, v->swapchain, nullptr);
        vkDestroyCommandPool(v->device, v->commandPool, nullptr);
        vkDestroyDevice(v->device, nullptr);
    }

    if (v->surface && v->instance)
        vkDestroySurfaceKHR(v->instance, v->surface, nullptr);
    if (v->instance)
        vkDestroyInstance(v->instance, nullptr);

    if (v->window) glfwDestroyWindow(v->window);
    glfwTerminate();

    delete v;
}

}  // extern "C"
