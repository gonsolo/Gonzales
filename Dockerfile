FROM archlinux:latest

# Update and install system dependencies
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    base-devel \
    git \
    cmake \
    ninja \
    openimageio \
    ptex \
    wget \
    unzip \
    sudo \
    embree \
    fmt

# Setup builder user for AUR
RUN useradd -m builder && \
    echo "builder ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/builder && \
    chmod 0440 /etc/sudoers.d/builder

# Install Swift from AUR
RUN su - builder -c "git clone https://aur.archlinux.org/swift-bin.git && cd swift-bin && makepkg -si --noconfirm"

# Install SwiftLint from AUR
RUN su - builder -c "git clone https://aur.archlinux.org/swiftlint.git && cd swiftlint && makepkg -si --noconfirm"

# Patch Swift's _CStdlib.h to fix cmath redefinition error
# Wraps the math.h include in #if 0 ... #endif
RUN sed -i 's/#if __has_include(<math.h>)/#if 0\n#if __has_include(<math.h>)/' /usr/lib/swift/lib/swift/_FoundationCShims/_CStdlib.h && \
    sed -i '/#include <math.h>/{n;s/#endif/#endif\n#endif/;}' /usr/lib/swift/lib/swift/_FoundationCShims/_CStdlib.h

# Cleanup
RUN pacman -Sc --noconfirm && \
    rm -rf /home/builder/swift-bin /home/builder/swiftlint
