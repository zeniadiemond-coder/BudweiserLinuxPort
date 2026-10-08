#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

inline int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            g_failures++;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_MSG(cond, msg)                                                     \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s (%s)\n", __FILE__, __LINE__, #cond, std::string(msg).c_str()); \
            g_failures++;                                                        \
        }                                                                        \
    } while (0)

inline const char* reference_dir() {
    const char* root = std::getenv("OPENV6_REFERENCE");
    return root;
}
