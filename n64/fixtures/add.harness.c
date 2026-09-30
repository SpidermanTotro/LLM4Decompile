#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
uint32_t func0(uint32_t a, uint32_t b);

int main(void) {
    const uint32_t cases[][3] = {
        {0, 0, 0}, {2, 3, 5}, {0xffffffffu, 1, 0},
        {0x80000000u, 0x80000000u, 0}, {0x12345678u, 0x11111111u, 0x23456789u}
    };
    for (unsigned int i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        uint32_t result = func0(cases[i][0], cases[i][1]);
        if (result != cases[i][2]) {
            fprintf(stderr, "case %u failed\n", i);
            return 1;
        }
        printf("%08" PRIx32 "\n", result);
    }
    return 0;
}
