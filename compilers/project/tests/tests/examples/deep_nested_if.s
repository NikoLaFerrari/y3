    .data
    .text
    .globl main

classify_number:
    pushq %rbp
    movq %rsp, %rbp
    subq $240, %rsp
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
    testq %rax, %rax
    jnz .L0
    jmp .L1
.L0:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq $100, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    movq -40(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -48(%rbp)
    movq -48(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq $1000, %rax
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -72(%rbp)
    movq -72(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq $4, %rax
    movq %rax, -80(%rbp)
    movq -80(%rbp), %rax
    leave
    ret
    jmp .L8
.L7:
    movq $3, %rax
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    leave
    ret
.L8:
    jmp .L5
.L4:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -96(%rbp)
    movq $10, %rax
    movq %rax, -104(%rbp)
    movq -96(%rbp), %rax
    movq -104(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -112(%rbp)
    movq -112(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq $2, %rax
    movq %rax, -120(%rbp)
    movq -120(%rbp), %rax
    leave
    ret
    jmp .L11
.L10:
    movq $1, %rax
    movq %rax, -128(%rbp)
    movq -128(%rbp), %rax
    leave
    ret
.L11:
.L5:
    jmp .L2
.L1:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -136(%rbp)
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
    jnz .L12
    jmp .L13
.L12:
    movq $0, %rax
    movq %rax, -160(%rbp)
    movq -160(%rbp), %rax
    leave
    ret
    jmp .L14
.L13:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq $-100, %rax
    movq %rax, -176(%rbp)
    movq -168(%rbp), %rax
    movq -176(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -184(%rbp)
    movq -184(%rbp), %rax
    testq %rax, %rax
    jnz .L15
    jmp .L16
.L15:
    movq $-3, %rax
    movq %rax, -192(%rbp)
    movq -192(%rbp), %rax
    leave
    ret
    jmp .L17
.L16:
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -200(%rbp)
    movq $-10, %rax
    movq %rax, -208(%rbp)
    movq -200(%rbp), %rax
    movq -208(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -216(%rbp)
    movq -216(%rbp), %rax
    testq %rax, %rax
    jnz .L18
    jmp .L19
.L18:
    movq $-2, %rax
    movq %rax, -224(%rbp)
    movq -224(%rbp), %rax
    leave
    ret
    jmp .L20
.L19:
    movq $-1, %rax
    movq %rax, -232(%rbp)
    movq -232(%rbp), %rax
    leave
    ret
.L20:
.L17:
.L14:
.L2:
    movq $0, %rax
    movq %rax, -240(%rbp)
    movq -240(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $128, %rsp
    movq $5, %rax
    movq %rax, -8(%rbp)
    leaq classify_number(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -16(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -16(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $50, %rax
    movq %rax, -24(%rbp)
    leaq classify_number(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -24(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -32(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $500, %rax
    movq %rax, -40(%rbp)
    leaq classify_number(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
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
    movq $5000, %rax
    movq %rax, -56(%rbp)
    leaq classify_number(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -56(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -64(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -64(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    movq %rax, -72(%rbp)
    leaq classify_number(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -72(%rbp), %rdi
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
    movq $-5, %rax
    movq %rax, -88(%rbp)
    leaq classify_number(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -88(%rbp), %rdi
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
    movq $-50, %rax
    movq %rax, -104(%rbp)
    leaq classify_number(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -104(%rbp), %rdi
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
    movq $-500, %rax
    movq %rax, -120(%rbp)
    leaq classify_number(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -120(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -128(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -128(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
