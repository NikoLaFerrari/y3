    .data
    .text
    .globl main

square:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    leave
    ret

add_three:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    leave
    ret

max:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
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
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq -32(%rbp), %rax
    leave
    ret
    jmp .L2
.L1:
.L2:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -40(%rbp), %rax
    leave
    ret

min:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq -32(%rbp), %rax
    leave
    ret
    jmp .L5
.L4:
.L5:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq -40(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $256, %rsp
    movq $2, %rax
    movq %rax, -8(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -16(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
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
    movq $2, %rax
    movq %rax, -32(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -40(%rbp)
    movq $3, %rax
    movq %rax, -48(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -48(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -56(%rbp)
    movq $4, %rax
    movq %rax, -64(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -72(%rbp)
    leaq add_three(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    movq -56(%rbp), %rsi
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
    movq $5, %rax
    movq %rax, -88(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -96(%rbp)
    movq $4, %rax
    movq %rax, -104(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -104(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -112(%rbp)
    leaq max(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -96(%rbp), %rdi
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
    movq $5, %rax
    movq %rax, -128(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -128(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -136(%rbp)
    movq $4, %rax
    movq %rax, -144(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -144(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -152(%rbp)
    leaq min(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -136(%rbp), %rdi
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
    movq $3, %rax
    movq %rax, -168(%rbp)
    movq $7, %rax
    movq %rax, -176(%rbp)
    leaq max(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -168(%rbp), %rdi
    movq -176(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -184(%rbp)
    leaq square(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -184(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -192(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -192(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $10, %rax
    movq %rax, -200(%rbp)
    movq $5, %rax
    movq %rax, -208(%rbp)
    leaq min(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -200(%rbp), %rdi
    movq -208(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -216(%rbp)
    movq $20, %rax
    movq %rax, -224(%rbp)
    movq $8, %rax
    movq %rax, -232(%rbp)
    leaq min(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -224(%rbp), %rdi
    movq -232(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -240(%rbp)
    leaq max(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -216(%rbp), %rdi
    movq -240(%rbp), %rsi
    call *%rax
    addq $16, %rsp
    movq %rax, -248(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -248(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
