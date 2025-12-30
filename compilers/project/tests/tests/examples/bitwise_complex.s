    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $304, %rsp
    movq $12, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $25, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    andq %rcx, %rax
    movq %rax, -56(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq -64(%rbp), %rax
    movq -72(%rbp), %rcx
    orq %rcx, %rax
    movq %rax, -80(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -88(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq -88(%rbp), %rax
    movq -96(%rbp), %rcx
    xorq %rcx, %rax
    movq %rax, -104(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -104(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -112(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq -112(%rbp), %rax
    movq -120(%rbp), %rcx
    andq %rcx, %rax
    movq %rax, -128(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -136(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -144(%rbp)
    movq -136(%rbp), %rax
    movq -144(%rbp), %rcx
    xorq %rcx, %rax
    movq %rax, -152(%rbp)
    movq -128(%rbp), %rax
    movq -152(%rbp), %rcx
    orq %rcx, %rax
    movq %rax, -160(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -160(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    orq %rcx, %rax
    movq %rax, -184(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -192(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq -192(%rbp), %rax
    movq -200(%rbp), %rcx
    xorq %rcx, %rax
    movq %rax, -208(%rbp)
    movq -184(%rbp), %rax
    movq -208(%rbp), %rcx
    andq %rcx, %rax
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
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -240(%rbp)
    movq -232(%rbp), %rax
    movq -240(%rbp), %rcx
    andq %rcx, %rax
    movq %rax, -248(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -248(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -256(%rbp)
    movq $1, %rax
    movq %rax, -264(%rbp)
    movq -256(%rbp), %rax
    movq -264(%rbp), %rcx
    shlq %cl, %rax
    movq %rax, -272(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -272(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -280(%rbp)
    movq $2, %rax
    movq %rax, -288(%rbp)
    movq -280(%rbp), %rax
    movq -288(%rbp), %rcx
    sarq %cl, %rax
    movq %rax, -296(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -296(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
