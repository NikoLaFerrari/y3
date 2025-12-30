    .data
    .text
    .globl main

test:
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq $5, %rax
    movq %rax, -8(%rbp)
    movq -8(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq $10, %rax
    movq %rax, -16(%rbp)
    movq %rbp, %rax
    movq -16(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
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
