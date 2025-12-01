#include <stdio.h>
#include <stdint.h>

void _bx_print_int(int64_t x) {
    printf("%ld\n", x);
}

void _bx_print_bool(int64_t b) {
    printf(b == 0 ? "false\n" : "true\n");
}
