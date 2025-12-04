#include <stdio.h>

void __bx_print_int(long val) {
    printf("%ld\n", val);
}

void __bx_print_bool(long val) {
    if (val == 0) {
        printf("false\n");
    } else {
        printf("true\n");
    }
}
