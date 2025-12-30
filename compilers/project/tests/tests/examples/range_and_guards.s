    .data
    .text
    .globl main

is_in_range:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    cmpq %rcx, %rax
    setge %al
    movzbq %al, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    cmpq %rcx, %rax
    setle %al
    movzbq %al, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq $1, %rax
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rax
    leave
    ret
    jmp .L5
.L4:
.L5:
    jmp .L2
.L1:
.L2:
    movq $0, %rax
    movq %rax, -64(%rbp)
    movq -64(%rbp), %rax
    leave
    ret

safe_divide:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $0, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    cmpq %rcx, %rax
    setne %al
    movzbq %al, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    leave
    ret
    jmp .L8
.L7:
.L8:
    movq $-1, %rax
    movq %rax, -56(%rbp)
    movq -56(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $288, %rsp
    movq $5, %rax
    movq %rax, -24(%rbp)
    movq $1, %rax
    movq %rax, -32(%rbp)
    movq $10, %rax
    movq %rax, -40(%rbp)
    leaq is_in_range(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    movq -32(%rbp), %rsi
    movq -40(%rbp), %rdx
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
    movq $15, %rax
    movq %rax, -56(%rbp)
    movq $1, %rax
    movq %rax, -64(%rbp)
    movq $10, %rax
    movq %rax, -72(%rbp)
    leaq is_in_range(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    movq -64(%rbp), %rsi
    movq -72(%rbp), %rdx
    call *%rax
    addq $16, %rsp
    movq %rax, -80(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    movq %rax, -88(%rbp)
    movq $1, %rax
    movq %rax, -96(%rbp)
    movq $10, %rax
    movq %rax, -104(%rbp)
    leaq is_in_range(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    movq -96(%rbp), %rsi
    movq -104(%rbp), %rdx
    call *%rax
    addq $16, %rsp
    movq %rax, -112(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -112(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $20, %rax
    movq %rax, -120(%rbp)
    movq $4, %rax
    movq %rax, -128(%rbp)
    leaq safe_divide(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    movq -128(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -136(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -136(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $100, %rax
    movq %rax, -144(%rbp)
    movq $5, %rax
    movq %rax, -152(%rbp)
    leaq safe_divide(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -144(%rbp), %rdi
    movq -152(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -160(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -160(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $10, %rax
    movq %rax, -168(%rbp)
    movq $0, %rax
    movq %rax, -176(%rbp)
    leaq safe_divide(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -168(%rbp), %rdi
    movq -176(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -184(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -184(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $7, %rax
    movq %rax, -192(%rbp)
    movq %rbp, %rax
    movq -192(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq $5, %rax
    movq %rax, -208(%rbp)
    movq $10, %rax
    movq %rax, -216(%rbp)
    leaq is_in_range(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -200(%rbp), %rdi
    movq -208(%rbp), %rsi
    movq -216(%rbp), %rdx
    call *%rax
    addq $16, %rsp
    movq %rax, -224(%rbp)
    movq %rbp, %rax
    movq -224(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -232(%rbp)
    movq $1, %rax
    movq %rax, -240(%rbp)
    movq -232(%rbp), %rax
    movq -240(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -248(%rbp)
    movq -248(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -256(%rbp)
    movq $2, %rax
    movq %rax, -264(%rbp)
    leaq safe_divide(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -256(%rbp), %rdi
    movq -264(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -272(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -272(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L11
.L10:
    movq $-999, %rax
    movq %rax, -280(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -280(%rbp), %rdi
    call *%rax
    addq $16, %rsp
.L11:
    movq $0, %rax
    leave
    ret
