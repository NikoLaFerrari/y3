    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $160, %rsp
    movq $837799, %rax
    movq %rax, -8(%rbp)
    movq %rbp, %rax
    movq -8(%rbp), %rcx
    movq %rcx, -8(%rax)
.L0:
    movq $1, %rax
    movq %rax, -16(%rbp)
    movq -16(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -24(%rbp), %rdi
    call *%rax
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq $1, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    jmp .L2
    jmp .L5
.L4:
.L5:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq $2, %rax
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -72(%rbp)
    movq $0, %rax
    movq %rax, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq $2, %rax
    movq %rax, -104(%rbp)
    movq -96(%rbp), %rax
    movq -104(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rax, -112(%rbp)
    movq %rbp, %rax
    movq -112(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L8
.L7:
    movq $3, %rax
    movq %rax, -120(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq -120(%rbp), %rax
    movq -128(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -136(%rbp)
    movq $1, %rax
    movq %rax, -144(%rbp)
    movq -136(%rbp), %rax
    movq -144(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -152(%rbp)
    movq %rbp, %rax
    movq -152(%rbp), %rcx
    movq %rcx, -8(%rax)
.L8:
    jmp .L0
.L2:
    movq $0, %rax
    leave
    ret
