    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $96, %rsp
    movq %r10, -8(%rbp)
    movq $74, %rax
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rcx
    movq %rcx, -16(%rbp)
    movq $13, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rcx
    movq %rcx, -24(%rbp)
    movq $0, %rax
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rcx
    movq %rcx, -32(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -64(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, -72(%rbp)
    movq -64(%rbp), %rax
    movq -72(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -80(%rbp)
    movq -80(%rbp), %rcx
    movq %rcx, -32(%rbp)
    movq -32(%rbp), %rcx
    movq %rcx, -88(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -88(%rbp), %rdi
    call *%rax
    movq $0, %rax
    leave
    ret
