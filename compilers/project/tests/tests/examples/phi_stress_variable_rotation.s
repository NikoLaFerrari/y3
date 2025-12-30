    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $512, %rsp
    movq $10, %rax
    movq %rax, -48(%rbp)
    movq %rbp, %rax
    movq -48(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $20, %rax
    movq %rax, -56(%rbp)
    movq %rbp, %rax
    movq -56(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq $30, %rax
    movq %rax, -64(%rbp)
    movq %rbp, %rax
    movq -64(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq $0, %rax
    movq %rax, -72(%rbp)
    movq %rbp, %rax
    movq -72(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq $0, %rax
    movq %rax, -80(%rbp)
    movq %rbp, %rax
    movq -80(%rbp), %rcx
    movq %rcx, -40(%rax)
.L0:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -88(%rbp)
    movq $10, %rax
    movq %rax, -96(%rbp)
    movq -88(%rbp), %rax
    movq -96(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -104(%rbp)
    movq -104(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -112(%rbp)
    movq $3, %rax
    movq %rax, -120(%rbp)
    movq -112(%rbp), %rax
    movq -120(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -128(%rbp)
    movq $0, %rax
    movq %rax, -136(%rbp)
    movq -128(%rbp), %rax
    movq -136(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -144(%rbp)
    movq -144(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -152(%rbp)
    movq %rbp, %rax
    movq -152(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -160(%rbp)
    movq %rbp, %rax
    movq -160(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -168(%rbp)
    movq %rbp, %rax
    movq -168(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -176(%rbp)
    movq %rbp, %rax
    movq -176(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L5
.L4:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -184(%rbp)
    movq $3, %rax
    movq %rax, -192(%rbp)
    movq -184(%rbp), %rax
    movq -192(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -200(%rbp)
    movq $1, %rax
    movq %rax, -208(%rbp)
    movq -200(%rbp), %rax
    movq -208(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -216(%rbp)
    movq -216(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -224(%rbp)
    movq %rbp, %rax
    movq -224(%rbp), %rcx
    movq %rcx, -32(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -232(%rbp)
    movq %rbp, %rax
    movq -232(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -240(%rbp)
    movq %rbp, %rax
    movq -240(%rbp), %rcx
    movq %rcx, -24(%rax)
    movq %rbp, %rax
    movq -32(%rax), %rcx
    movq %rcx, -248(%rbp)
    movq %rbp, %rax
    movq -248(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L8
.L7:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -256(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -264(%rbp)
    movq -256(%rbp), %rax
    movq -264(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -272(%rbp)
    movq %rbp, %rax
    movq -272(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -280(%rbp)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -288(%rbp)
    movq -280(%rbp), %rax
    movq -288(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -296(%rbp)
    movq %rbp, %rax
    movq -296(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -304(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -312(%rbp)
    movq -304(%rbp), %rax
    movq -312(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -320(%rbp)
    movq %rbp, %rax
    movq -320(%rbp), %rcx
    movq %rcx, -24(%rax)
.L8:
.L5:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -328(%rbp)
    movq $100, %rax
    movq %rax, -336(%rbp)
    movq -328(%rbp), %rax
    movq -336(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -344(%rbp)
    movq -344(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -352(%rbp)
    movq $50, %rax
    movq %rax, -360(%rbp)
    movq -352(%rbp), %rax
    movq -360(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -368(%rbp)
    movq %rbp, %rax
    movq -368(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L11
.L10:
.L11:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -376(%rbp)
    movq $100, %rax
    movq %rax, -384(%rbp)
    movq -376(%rbp), %rax
    movq -384(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -392(%rbp)
    movq -392(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -400(%rbp)
    movq $50, %rax
    movq %rax, -408(%rbp)
    movq -400(%rbp), %rax
    movq -408(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -416(%rbp)
    movq %rbp, %rax
    movq -416(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L14
.L13:
.L14:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -424(%rbp)
    movq $100, %rax
    movq %rax, -432(%rbp)
    movq -424(%rbp), %rax
    movq -432(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -440(%rbp)
    movq -440(%rbp), %rax
    testq %rax, %rax
    jnz .L15
    jmp .L16
.L15:
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -448(%rbp)
    movq $50, %rax
    movq %rax, -456(%rbp)
    movq -448(%rbp), %rax
    movq -456(%rbp), %rcx
    subq %rcx, %rax
    movq %rax, -464(%rbp)
    movq %rbp, %rax
    movq -464(%rbp), %rcx
    movq %rcx, -24(%rax)
    jmp .L17
.L16:
.L17:
    movq %rbp, %rax
    movq -40(%rax), %rcx
    movq %rcx, -472(%rbp)
    movq $1, %rax
    movq %rax, -480(%rbp)
    movq -472(%rbp), %rax
    movq -480(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -488(%rbp)
    movq %rbp, %rax
    movq -488(%rbp), %rcx
    movq %rcx, -40(%rax)
    jmp .L0
.L2:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -496(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -496(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -504(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -504(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rbp, %rax
    movq -24(%rax), %rcx
    movq %rcx, -512(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -512(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
