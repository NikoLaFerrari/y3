    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $256, %rsp
    movq $0, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $0, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $5, %rax
    movq %rax, -40(%rbp)
    movq %rbp, %rax
    movq -40(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $0, %rax
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq -48(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq $0, %rax
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -72(%rbp)
    movq -72(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq $10, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -88(%rbp)
    movq $1, %rax
    movq %rax, -96(%rbp)
    movq -88(%rbp), %rax
    movq -96(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -104(%rbp)
    movq -104(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq $20, %rax
    movq %rax, -112(%rbp)
    movq %rbp, %rax
    movq -112(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L5
.L4:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq $2, %rax
    movq %rax, -128(%rbp)
    movq -120(%rbp), %rax
    movq -128(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -136(%rbp)
    movq -136(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq $30, %rax
    movq %rax, -144(%rbp)
    movq %rbp, %rax
    movq -144(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L8
.L7:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq $3, %rax
    movq %rax, -160(%rbp)
    movq -152(%rbp), %rax
    movq -160(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -168(%rbp)
    movq -168(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq $40, %rax
    movq %rax, -176(%rbp)
    movq %rbp, %rax
    movq -176(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L11
.L10:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -184(%rbp)
    movq $4, %rax
    movq %rax, -192(%rbp)
    movq -184(%rbp), %rax
    movq -192(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -200(%rbp)
    movq -200(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq $50, %rax
    movq %rax, -208(%rbp)
    movq %rbp, %rax
    movq -208(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L14
.L13:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -216(%rbp)
    movq $5, %rax
    movq %rax, -224(%rbp)
    movq -216(%rbp), %rax
    movq -224(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -232(%rbp)
    movq -232(%rbp), %rax
    testq %rax, %rax
    jnz .L15
    jmp .L16
.L15:
    movq $60, %rax
    movq %rax, -240(%rbp)
    movq %rbp, %rax
    movq -240(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L17
.L16:
    movq $70, %rax
    movq %rax, -248(%rbp)
    movq %rbp, %rax
    movq -248(%rbp), %rcx
    movq %rcx, -16(%rax)
.L17:
.L14:
.L11:
.L8:
.L5:
.L2:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -256(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -256(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
