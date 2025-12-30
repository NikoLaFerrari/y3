    .data
    .globl glob_x
glob_x:
    .zero 8
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq glob_x(%rip), %rcx
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rax
    notq %rax
    movq %rax, -16(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, glob_x(%rip)
    movq glob_x(%rip), %rcx
    movq %rcx, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
