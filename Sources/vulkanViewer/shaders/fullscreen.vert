#version 450

layout(location = 0) out vec2 fragTexCoord;

void main() {
    // Generate fullscreen triangle from gl_VertexIndex (0, 1, 2)
    // No vertex buffer needed — positions computed from vertex ID
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
    fragTexCoord = vec2(pos.x, pos.y);
}
