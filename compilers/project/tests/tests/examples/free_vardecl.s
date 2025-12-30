    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $160, %rsp
    movq $10, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -48(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq $20, %rax
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -80(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -88(%rbp)
    movq $42, %rax
    movq %rax, -96(%rbp)
    movq -88(%rbp), %rax
    movq -96(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -104(%rbp)
    movq %rbp, %rax
    movq -104(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -112(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -112(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq -120(%rbp), %rax
    movq -128(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -136(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -144(%rbp)
    movq -136(%rbp), %rax
    movq -144(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -152(%rbp)
    movq %rbp, %rax
    movq -152(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -160(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -160(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
