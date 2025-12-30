    .data
    .text
    .globl main

is_positive:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $0, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    leave
    ret

is_even:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -8(%rbp)
    movq $2, %rax
    movq %rax, -16(%rbp)
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -24(%rbp)
    movq $0, %rax
    movq %rax, -32(%rbp)
    movq -24(%rbp), %rax
    movq -32(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $160, %rsp
    movq $10, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $-5, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -40(%rbp)
    leaq is_positive(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -56(%rbp)
    leaq is_even(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -64(%rbp)
    movq -64(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq $1, %rax
    movq %rax, -72(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L5
.L4:
.L5:
    jmp .L2
.L1:
.L2:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -80(%rbp)
    leaq is_positive(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -80(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -96(%rbp)
    leaq is_even(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -96(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -104(%rbp)
    movq -104(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq $2, %rax
    movq %rax, -112(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -112(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L11
.L10:
.L11:
    jmp .L8
.L7:
.L8:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -120(%rbp)
    leaq is_positive(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -128(%rbp)
    movq -128(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq $3, %rax
    movq %rax, -136(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -136(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L14
.L13:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -144(%rbp)
    leaq is_even(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -144(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -152(%rbp)
    movq -152(%rbp), %rax
    testq %rax, %rax
    jnz .L15
    jmp .L16
.L15:
    movq $4, %rax
    movq %rax, -160(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -160(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L17
.L16:
.L17:
.L14:
    movq $0, %rax
    leave
    ret
