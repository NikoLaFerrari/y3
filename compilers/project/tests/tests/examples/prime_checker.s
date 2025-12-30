    .data
    .text
    .globl main

is_prime:
    pushq %rbp
    movq %rsp, %rbp
    subq $176, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq $2, %rax
    movq %rax, -24(%rbp)
    movq -16(%rbp), %rax
    movq -24(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -32(%rbp)
    movq -32(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq $0, %rax
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    leave
    ret
    jmp .L2
.L1:
.L2:
    movq $2, %rax
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq -48(%rbp), %rcx
    movq %rcx, -8(%rax)
.L3:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    cmpq %rcx, %rax
    setle %al
    movzbq %al, %rax
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    testq %rax, %rax
    jnz .L4
    jmp .L5
.L4:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq -96(%rbp), %rax
    movq -104(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -112(%rbp)
    movq $0, %rax
    movq %rax, -120(%rbp)
    movq -112(%rbp), %rax
    movq -120(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -128(%rbp)
    movq -128(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq $0, %rax
    movq %rax, -136(%rbp)
    movq -136(%rbp), %rax
    leave
    ret
    jmp .L8
.L7:
.L8:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -144(%rbp)
    movq $1, %rax
    movq %rax, -152(%rbp)
    movq -144(%rbp), %rax
    movq -152(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -160(%rbp)
    movq %rbp, %rax
    movq -160(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L3
.L5:
    movq $1, %rax
    movq %rax, -168(%rbp)
    movq -168(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $96, %rsp
    movq $2, %rax
    movq %rax, -16(%rbp)
    movq %rbp, %rax
    movq -16(%rbp), %rcx
    movq %rcx, -8(%rax)
.L9:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -24(%rbp)
    movq $20, %rax
    movq %rax, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    cmpq %rcx, %rax
    setle %al
    movzbq %al, %rax
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    testq %rax, %rax
    jnz .L10
    jmp .L11
.L10:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -48(%rbp)
    leaq is_prime(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L14
.L13:
.L14:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq $1, %rax
    movq %rax, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -88(%rbp)
    movq %rbp, %rax
    movq -88(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L9
.L11:
    movq $0, %rax
    leave
    ret
