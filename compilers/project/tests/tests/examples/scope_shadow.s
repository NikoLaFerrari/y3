    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    movq $10, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $20, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq $1, %rax
    movq %rax, -56(%rbp)
    movq -48(%rbp), %rax
    movq -56(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -72(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
