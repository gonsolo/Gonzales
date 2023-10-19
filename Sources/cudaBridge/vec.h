#pragma once

#if defined(__CUDACC__)
#define __gdt_device __device__
#define __gdt_host __host__
#else
#define __gdt_device /* nix */
#define __gdt_host   /* nix */
#endif

#define __both__ __gdt_host __gdt_device

template <typename T> struct vec3 {

        inline __both__ vec3() {}

        inline __both__ vec3(T v) {
                x = v;
                y = v;
                z = v;
        }

        inline __both__ vec3(T x, T y, T z) {
                this->x = x;
                this->y = y;
                this->z = z;
        }

        inline __both__ friend vec3 operator+(vec3 lhs, const vec3 &rhs) {
                lhs.x = lhs.x + rhs.x;
                lhs.y = lhs.y + rhs.y;
                lhs.z = lhs.z + rhs.z;
                return lhs;
        }

        inline __both__ friend vec3 operator-(vec3 lhs, const vec3 &rhs) {
                lhs.x = lhs.x - rhs.x;
                lhs.y = lhs.y - rhs.y;
                lhs.z = lhs.z - rhs.z;
                return lhs;
        }

        inline __both__ friend vec3 operator+(T lhs, const vec3 &rhs) {
                vec3 result;
                result.x = lhs + rhs.x;
                result.y = lhs + rhs.y;
                result.z = lhs + rhs.z;
                return result;
        }

        inline __both__ friend vec3 operator+(const vec3 &lhs, T rhs) { return rhs + lhs; }

        inline __both__ friend vec3 operator*(T lhs, const vec3 &rhs) {
                vec3 result;
                result.x = lhs * rhs.x;
                result.y = lhs * rhs.y;
                result.z = lhs * rhs.z;
                return result;
        }

        inline __both__ friend vec3 operator*(const vec3 &lhs, T rhs) { return rhs * lhs; }

        inline __both__ friend vec3 operator/(const vec3 &lhs, T rhs) {
                vec3 result;
                result.x = lhs.x / rhs;
                result.y = lhs.y / rhs;
                result.z = lhs.z / rhs;
                return result;
        }

        T x;
        T y;
        T z;
};

template <typename T> struct vec2 {

        inline __both__ vec2() {}

        inline __both__ vec2(T x, T y) {
                this->x = x;
                this->y = y;
        }

        T x;
        T y;
};

template <typename T> inline __both__ T dot(const vec3<T> &a, const vec3<T> &b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T> inline __both__ vec3<T> normalize(const vec3<T> &v) {
        return v * 1.f / sqrt(dot(v, v));
}

template <typename T> inline __both__ vec3<T> cross(const vec3<T> &a, const vec3<T> &b) {
        return vec3<T>(a.y * b.z - b.y * a.z, a.z * b.x - b.z * a.x, a.x * b.y - b.x * a.y);
}

using vec3f = vec3<float>;
using vec3i = vec3<int>;
using vec2f = vec2<float>;
using vec2i = vec2<int>;
