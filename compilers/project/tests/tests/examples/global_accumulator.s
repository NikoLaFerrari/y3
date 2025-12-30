    .data
    .globl glob_accumulator
glob_accumulator:
    .zero 8
    .globl glob_multiplier
glob_multiplier:
    .zero 8
    .text
    .globl main

add_five:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_accumulator(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq $5, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_accumulator(%rip)
    leave
    ret

subtract_three:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_accumulator(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq $3, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_accumulator(%rip)
    leave
    ret

apply_multiplier:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_accumulator(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq glob_multiplier(%rip), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, glob_accumulator(%rip)
    leave
    ret

set_multiplier:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rcx
    movq %rcx, glob_multiplier(%rip)
    leave
    ret

complex_calculation:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    leaq add_five(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq add_five(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq subtract_three(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq $2, %rax
    movq %rax, -8(%rbp)
    leaq set_multiplier(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq apply_multiplier(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leaq subtract_three(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_accumulator(%rip), %rcx
    movq %rcx, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_multiplier(%rip), %rcx
    movq %rcx, -16(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    leaq complex_calculation(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    call *%rax
    addq $16, %rsp
    movq glob_accumulator(%rip), %rcx
    movq %rcx, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq glob_multiplier(%rip), %rcx
    movq %rcx, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
