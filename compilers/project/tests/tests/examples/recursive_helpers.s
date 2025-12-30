    .data
    .text
    .globl main

sum_range_recursive:
    pushq %rbp
    movq %rsp, %rbp
    subq $96, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq $0, %rax
    movq %rax, -32(%rbp)
    movq -32(%rbp), %rax
    leave
    ret
    jmp .L2
.L1:
.L2:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq $1, %rax
    movq %rax, -56(%rbp)
    movq -48(%rbp), %rax
    movq -56(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -72(%rbp)
    leaq sum_range_recursive(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    movq -72(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -80(%rbp)
    movq -40(%rbp), %rax
    movq -80(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    leave
    ret

power:
    pushq %rbp
    movq %rsp, %rbp
    subq $128, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $0, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq $1, %rax
    movq %rax, -32(%rbp)
    movq -32(%rbp), %rax
    leave
    ret
    jmp .L5
.L4:
.L5:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq $1, %rax
    movq %rax, -48(%rbp)
    movq -40(%rbp), %rax
    movq -48(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -64(%rbp)
    movq -64(%rbp), %rax
    leave
    ret
    jmp .L8
.L7:
.L8:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -88(%rbp)
    movq $1, %rax
    movq %rax, -96(%rbp)
    movq -88(%rbp), %rax
    movq -96(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -104(%rbp)
    leaq power(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    movq -104(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -112(%rbp)
    movq -72(%rbp), %rax
    movq -112(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -120(%rbp)
    movq -120(%rbp), %rax
    leave
    ret

gcd:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $0, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq -32(%rbp), %rax
    leave
    ret
    jmp .L11
.L10:
.L11:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq -48(%rbp), %rax
    movq -56(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -64(%rbp)
    leaq gcd(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    movq -64(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -72(%rbp)
    movq -72(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $144, %rsp
    movq $1, %rax
    movq %rax, -8(%rbp)
    movq $5, %rax
    movq %rax, -16(%rbp)
    leaq sum_range_recursive(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    movq -16(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -24(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $1, %rax
    movq %rax, -32(%rbp)
    movq $10, %rax
    movq %rax, -40(%rbp)
    leaq sum_range_recursive(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    movq -40(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -48(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $2, %rax
    movq %rax, -56(%rbp)
    movq $5, %rax
    movq %rax, -64(%rbp)
    leaq power(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    movq -64(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -72(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $3, %rax
    movq %rax, -80(%rbp)
    movq $3, %rax
    movq %rax, -88(%rbp)
    leaq power(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    movq -88(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -96(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -96(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $48, %rax
    movq %rax, -104(%rbp)
    movq $18, %rax
    movq %rax, -112(%rbp)
    leaq gcd(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -104(%rbp), %rdi
    movq -112(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -120(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $100, %rax
    movq %rax, -128(%rbp)
    movq $35, %rax
    movq %rax, -136(%rbp)
    leaq gcd(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -128(%rbp), %rdi
    movq -136(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -144(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -144(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
