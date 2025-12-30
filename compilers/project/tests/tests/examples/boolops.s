    .data
    .text
    .globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $592, %rsp
    movq $10, %rax
    movq %rax, -24(%rbp)
    movq %rbp, %rax
    movq -24(%rbp), %rcx
    movq %rcx, -8(%rax)
    movq $20, %rax
    movq %rax, -32(%rbp)
    movq %rbp, %rax
    movq -32(%rbp), %rcx
    movq %rcx, -16(%rax)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -40(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -48(%rbp)
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
    movq $0, %rax
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
.L2:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    cmpq %rcx, %rax
    setne %al
    movzbq %al, %rax
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    testq %rax, %rax
    jnz .L3
    jmp .L4
.L3:
    movq $1, %rax
    movq %rax, -96(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -96(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L5
.L4:
.L5:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq $2, %rax
    movq %rax, -112(%rbp)
    movq -104(%rbp), %rax
    movq -112(%rbp), %rcx
    imulq %rcx, %rax
    movq %rax, -120(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -128(%rbp)
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
    movq $2, %rax
    movq %rax, -144(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -144(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L8
.L7:
.L8:
    movq $1, %rax
    movq %rax, -152(%rbp)
    movq $1, %rax
    movq %rax, -160(%rbp)
    movq -152(%rbp), %rax
    movq -160(%rbp), %rcx
    movq %rax, -168(%rbp)
    movq -168(%rbp), %rax
    testq %rax, %rax
    jnz .L9
    jmp .L10
.L9:
    movq $3, %rax
    movq %rax, -176(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -176(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L11
.L10:
.L11:
    movq $1, %rax
    movq %rax, -184(%rbp)
    movq $1, %rax
    movq %rax, -192(%rbp)
    movq -184(%rbp), %rax
    movq -192(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -200(%rbp)
    movq $0, %rax
    movq %rax, -208(%rbp)
    movq -200(%rbp), %rax
    movq -208(%rbp), %rcx
    movq %rax, -216(%rbp)
    movq -216(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    movq $4, %rax
    movq %rax, -224(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -224(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L14
.L13:
.L14:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -232(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -240(%rbp)
    movq -232(%rbp), %rax
    movq -240(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -248(%rbp)
    movq -248(%rbp), %rax
    testq %rax, %rax
    jnz .L15
    jmp .L16
.L15:
    movq $5, %rax
    movq %rax, -256(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -256(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L17
.L16:
.L17:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -264(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -272(%rbp)
    movq -264(%rbp), %rax
    movq -272(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -280(%rbp)
    movq -280(%rbp), %rax
    testq %rax, %rax
    jnz .L18
    jmp .L19
.L18:
    movq $6, %rax
    movq %rax, -288(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -288(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L20
.L19:
.L20:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -296(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -304(%rbp)
    movq -296(%rbp), %rax
    movq -304(%rbp), %rcx
    cmpq %rcx, %rax
    setl %al
    movzbq %al, %rax
    movq %rax, -312(%rbp)
    movq -312(%rbp), %rax
    testq %rax, %rax
    jnz .L21
    jmp .L22
.L21:
    movq $7, %rax
    movq %rax, -320(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -320(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L23
.L22:
.L23:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -328(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -336(%rbp)
    movq -328(%rbp), %rax
    movq -336(%rbp), %rcx
    cmpq %rcx, %rax
    setle %al
    movzbq %al, %rax
    movq %rax, -344(%rbp)
    movq -344(%rbp), %rax
    testq %rax, %rax
    jnz .L24
    jmp .L25
.L24:
    movq $8, %rax
    movq %rax, -352(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -352(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L26
.L25:
.L26:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -360(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -368(%rbp)
    movq -360(%rbp), %rax
    movq -368(%rbp), %rcx
    cmpq %rcx, %rax
    setge %al
    movzbq %al, %rax
    movq %rax, -376(%rbp)
    movq -376(%rbp), %rax
    testq %rax, %rax
    jnz .L27
    jmp .L28
.L27:
    movq $9, %rax
    movq %rax, -384(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -384(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L29
.L28:
.L29:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -392(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -400(%rbp)
    movq -392(%rbp), %rax
    movq -400(%rbp), %rcx
    cmpq %rcx, %rax
    setle %al
    movzbq %al, %rax
    movq %rax, -408(%rbp)
    movq -408(%rbp), %rax
    testq %rax, %rax
    jnz .L30
    jmp .L31
.L30:
    movq $10, %rax
    movq %rax, -416(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -416(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L32
.L31:
.L32:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -424(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -432(%rbp)
    movq -424(%rbp), %rax
    movq -432(%rbp), %rcx
    cmpq %rcx, %rax
    setle %al
    movzbq %al, %rax
    movq %rax, -440(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -448(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -456(%rbp)
    movq -448(%rbp), %rax
    movq -456(%rbp), %rcx
    cmpq %rcx, %rax
    setle %al
    movzbq %al, %rax
    movq %rax, -464(%rbp)
    movq -440(%rbp), %rax
    movq -464(%rbp), %rcx
    movq %rax, -472(%rbp)
    movq -472(%rbp), %rax
    testq %rax, %rax
    jnz .L33
    jmp .L34
.L33:
    movq $11, %rax
    movq %rax, -480(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -480(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L35
.L34:
.L35:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -488(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -496(%rbp)
    movq -488(%rbp), %rax
    movq -496(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -504(%rbp)
    movq -504(%rbp), %rax
    testq %rax, %rax
    jnz .L36
    jmp .L37
.L36:
    movq $12, %rax
    movq %rax, -512(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -512(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L38
.L37:
.L38:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -520(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -528(%rbp)
    movq -520(%rbp), %rax
    movq -528(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -536(%rbp)
    movq -536(%rbp), %rax
    xorq $1, %rax
    movq %rax, -544(%rbp)
    movq -544(%rbp), %rax
    testq %rax, %rax
    jnz .L39
    jmp .L40
.L39:
    movq $13, %rax
    movq %rax, -552(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -552(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L41
.L40:
.L41:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -560(%rbp)
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -568(%rbp)
    movq -560(%rbp), %rax
    movq -568(%rbp), %rcx
    cmpq %rcx, %rax
    setne %al
    movzbq %al, %rax
    movq %rax, -576(%rbp)
    movq -576(%rbp), %rax
    xorq $1, %rax
    movq %rax, -584(%rbp)
    movq -584(%rbp), %rax
    testq %rax, %rax
    jnz .L42
    jmp .L43
.L42:
    movq $14, %rax
    movq %rax, -592(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -592(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L44
.L43:
.L44:
    movq $0, %rax
    leave
    ret
