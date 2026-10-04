#include <cstdio>

int main() {
#if defined(__x86_64__)
    const char* arch = "x86_64";
#elif defined(__aarch64__)
    const char* arch = "aarch64";
#else
    const char* arch = "unknown";
#endif

#if defined(__clang__)
    std::printf("hello from %s (clang %d.%d.%d)\n", arch, __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    std::printf("hello from %s (gcc %d.%d.%d)\n", arch, __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    std::printf("hello from %s\n", arch);
#endif
    return 0;
}
