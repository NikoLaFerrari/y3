    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $144, %rsp
    movq %r10, -8(%rbp)
    movq $0, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rcx
    movq %rcx, -16(%rbp)
.L0:
    movq -16(%rbp), %rcx
    movq %rcx, -32(%rbp)
    movq $9, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq -16(%rbp), %rcx
    movq %rcx, -56(%rbp)
    movq $1, %rax
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -72(%rbp)
    movq -72(%rbp), %rcx
    movq %rcx, -16(%rbp)
    movq -16(%rbp), %rcx
    movq %rcx, -80(%rbp)
    movq $2, %rax
    movq %rax, -88(%rbp)
    movq -80(%rbp), %rax
    movq -88(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -96(%rbp)
    movq $0, %rax
    movq %rax, -104(%rbp)
    movq -96(%rbp), %rax
    movq -104(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -112(%rbp)
    movq -112(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    jmp .L0
    jmp .L5
.L4:
.L5:
    movq -16(%rbp), %rcx
    movq %rcx, -120(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    movq -120(%rbp), %rdi
    call *%rax
    movq -16(%rbp), %rcx
    movq %rcx, -128(%rbp)
    movq $7, %rax
    movq %rax, -136(%rbp)
    movq -128(%rbp), %rax
    movq -136(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -144(%rbp)
    movq -144(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    jmp .L2
    jmp .L8
.L7:
.L8:
    jmp .L0
.L2:
    movq $0, %rax
    leave
    ret
