    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $256, %rsp
    movq $0, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $0, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $0, %rax
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq -48(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $10, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $42, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $-100, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -88(%rbp)
    movq -80(%rbp), %rax
    movq -88(%rbp), %rcx
    andq %rcx, %rax
    movq %rax, -96(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -96(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -112(%rbp)
    movq -104(%rbp), %rax
    movq -112(%rbp), %rcx
    andq %rcx, %rax
    movq %rax, -120(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -136(%rbp)
    movq -128(%rbp), %rax
    movq -136(%rbp), %rcx
    orq %rcx, %rax
    movq %rax, -144(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -144(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -160(%rbp)
    movq -152(%rbp), %rax
    movq -160(%rbp), %rcx
    orq %rcx, %rax
    movq %rax, -168(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -168(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -184(%rbp)
    movq -176(%rbp), %rax
    movq -184(%rbp), %rcx
    xorq %rcx, %rax
    movq %rax, -192(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -192(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -208(%rbp)
    movq -200(%rbp), %rax
    movq -208(%rbp), %rcx
    xorq %rcx, %rax
    movq %rax, -216(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -216(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -224(%rbp)
    movq -224(%rbp), %rax
    notq %rax
    movq %rax, -232(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -232(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -240(%rbp)
    movq -240(%rbp), %rax
    notq %rax
    movq %rax, -248(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -248(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
