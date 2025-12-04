    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $128, %rsp
    movq $20, %rax
    movq %rax, -8(%rbp)
    movq %rbp, %rax
    movq -8(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $0, %rax
    movq %rax, -16(%rbp)
    movq %rbp, %rax
    movq -16(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $1, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $0, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -32(%rax)
.L0:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq $0, %rax
    movq %rax, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq $1, %rax
    movq %rax, -72(%rbp)
    movq -64(%rbp), %rax
    movq -72(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -88(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -88(%rbp), %rdi
    call *%rax
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq -96(%rbp), %rax
    movq -104(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -112(%rbp)
    movq %rbp, %rax
    movq -112(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq %rbp, %rax
    movq -120(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq %rbp, %rax
    movq -128(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L0
.L2:
    movq $0, %rax
    leave
    ret
