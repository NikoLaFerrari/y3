    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $256, %rsp
    movq $833779, %rax
    movq %rax, -16(%rbp)
    movq %rbp, %rax
    movq -16(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq $2, %rax
    movq %rax, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -40(%rbp)
    movq $0, %rax
    movq %rax, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq $2, %rax
    movq %rax, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq $3, %rax
    movq %rax, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -88(%rbp)
    movq $0, %rax
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
    movq $3, %rax
    movq %rax, -112(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -112(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L5
.L4:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq $5, %rax
    movq %rax, -128(%rbp)
    movq -120(%rbp), %rax
    movq -128(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -136(%rbp)
    movq $0, %rax
    movq %rax, -144(%rbp)
    movq -136(%rbp), %rax
    movq -144(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -152(%rbp)
    movq -152(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq $5, %rax
    movq %rax, -160(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -160(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L8
.L7:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq $7, %rax
    movq %rax, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -184(%rbp)
    movq $0, %rax
    movq %rax, -192(%rbp)
    movq -184(%rbp), %rax
    movq -192(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -200(%rbp)
    movq -200(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq $7, %rax
    movq %rax, -208(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -208(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L11
.L10:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -216(%rbp)
    movq $29, %rax
    movq %rax, -224(%rbp)
    movq -216(%rbp), %rax
    movq -224(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -232(%rbp)
    movq $0, %rax
    movq %rax, -240(%rbp)
    movq -232(%rbp), %rax
    movq -240(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -248(%rbp)
    movq -248(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq $29, %rax
    movq %rax, -256(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -256(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L14
.L13:
.L14:
.L11:
.L8:
.L5:
.L2:
    movq $0, %rax
    leave
    ret
