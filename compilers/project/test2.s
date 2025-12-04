    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq $42, %rax
    movq %rax, -8(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -8(%rbp), %rdi
    call *%rax
    movq $0, %rax
    leave
    ret
