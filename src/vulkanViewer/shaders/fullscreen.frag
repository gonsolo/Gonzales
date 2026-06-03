#version 450

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

void main() {
    vec3 hdr = texture(texSampler, fragTexCoord).rgb;

    // Simple Reinhard tone mapping
    vec3 mapped = hdr / (hdr + vec3(1.0));

    // sRGB gamma correction
    vec3 srgb = mix(
        mapped * 12.92,
        1.055 * pow(mapped, vec3(1.0 / 2.4)) - 0.055,
        step(vec3(0.0031308), mapped)
    );

    outColor = vec4(srgb, 1.0);
}
