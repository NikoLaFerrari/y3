    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $160, %rsp
    movq $20, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $0, %rax
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq -48(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $1, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $0, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -32(%rax)
.L0:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq $0, %rax
    movq %rax, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq $1, %rax
    movq %rax, -104(%rbp)
    movq -96(%rbp), %rax
    movq -104(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -112(%rbp)
    movq %rbp, %rax
    movq -112(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -120(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -136(%rbp)
    movq -128(%rbp), %rax
    movq -136(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -144(%rbp)
    movq %rbp, %rax
    movq -144(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq %rbp, %rax
    movq -152(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -160(%rbp)
    movq %rbp, %rax
    movq -160(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L0
.L2:
    movq $0, %rax
    leave
    ret
