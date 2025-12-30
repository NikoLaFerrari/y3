    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $224, %rsp
    movq $20, %rax
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
.L0:
    movq $1, %rax
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq $0, %rax
    movq %rax, -72(%rbp)
    movq -64(%rbp), %rax
    movq -72(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -80(%rbp)
    movq -80(%rbp), %rax
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
    movq %rcx, -88(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $1, %rax
    movq %rax, -96(%rbp)
    movq %rbp, %rax
    movq -96(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $0, %rax
    movq %rax, -104(%rbp)
    movq %rbp, %rax
    movq -104(%rbp), %rcx
    movq %rcx, -24(%rax)
.L6:
    movq $1, %rax
    movq %rax, -112(%rbp)
    movq -112(%rbp), %rax
    testq %rax, %rax
    jnz .L7
    jmp .L8
.L7:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -128(%rbp)
    movq -120(%rbp), %rax
    movq -128(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -136(%rbp)
    movq -136(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    jmp .L8
    jmp .L11
.L10:
.L11:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -144(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq -144(%rbp), %rax
    movq -152(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -160(%rbp)
    movq %rbp, %rax
    movq -160(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq $1, %rax
    movq %rax, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -184(%rbp)
    movq %rbp, %rax
    movq -184(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L6
.L8:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -192(%rbp)
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
    movq $1, %rax
    movq %rax, -208(%rbp)
    movq -200(%rbp), %rax
    movq -208(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -216(%rbp)
    movq %rbp, %rax
    movq -216(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L0
.L2:
    movq $0, %rax
    leave
    ret
