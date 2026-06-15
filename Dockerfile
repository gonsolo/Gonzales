FROM archlinux:latest

# Update and install system dependencies
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    base-devel \
    git \
    cmake \
    ninja \
    openimagedenoise \
    openimageio \
    ptex \
    wget \
    unzip \
    sudo \
    fmt \
    glfw-x11 \
    vulkan-headers \
    vulkan-icd-loader \
    inetutils

# Setup builder user for AUR
RUN useradd -m builder && \
    echo "builder ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/builder && \
    chmod 0440 /etc/sudoers.d/builder

# Cleanup
RUN pacman -Sc --noconfirm
