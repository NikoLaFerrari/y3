    .data
    .text
    .globl main

print_multiples_of_three:
    pushq %rbp
    movq %rsp, %rbp
    subq $128, %rsp
    movq $0, %rax
    movq %rax, -16(%rbp)
    movq %rbp, %rax
    movq -16(%rbp), %rcx
    movq %rcx, -8(%rax)
.L0:
    movq $1, %rax
    movq %rax, -24(%rbp)
    movq -24(%rbp), %rax
    testq %rax, %rax
    jnz .L1
    jmp .L2
.L1:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -32(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -40(%rbp)
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
    jmp .L2
    jmp .L5
.L4:
.L5:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq $3, %rax
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    movq -64(%rbp), %rcx
    cqto
    idivq %rcx
    movq %rdx, %rax
    movq %rax, -72(%rbp)
    movq $0, %rax
    movq %rax, -80(%rbp)
    movq -72(%rbp), %rax
    movq -80(%rbp), %rcx
    cmpq %rcx, %rax
    sete %al
    movzbq %al, %rax
    movq %rax, -88(%rbp)
    movq -88(%rbp), %rax
    testq %rax, %rax
    jnz .L6
    jmp .L7
.L6:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -96(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -96(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    jmp .L8
.L7:
.L8:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -104(%rbp)
    movq $1, %rax
    movq %rax, -112(%rbp)
    movq -104(%rbp), %rax
    movq -112(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -120(%rbp)
    movq %rbp, %rax
    movq -120(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L0
.L2:
    leave
    ret

sum_even_numbers:
    pushq %rbp
    movq %rsp, %rbp
    subq $160, %rsp
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
.L9:
    movq $1, %rax
    movq %rax, -40(%rbp)
    movq -40(%rbp), %rax
    testq %rax, %rax
    jnz .L10
    jmp .L11
.L10:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -48(%rbp)
    movq %rbp, %rax
    movq 0(%rax), %rcx
    movq %rcx, -56(%rbp)
    movq -48(%rbp), %rax
    movq -56(%rbp), %rcx
    cmpq %rcx, %rax
    setg %al
    movzbq %al, %rax
    movq %rax, -64(%rbp)
    movq -64(%rbp), %rax
    testq %rax, %rax
    jnz .L12
    jmp .L13
.L12:
    jmp .L11
    jmp .L14
.L13:
.L14:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -72(%rbp)
    movq $2, %rax
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
    jnz .L15
    jmp .L16
.L15:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -112(%rbp)
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -120(%rbp)
    movq -112(%rbp), %rax
    movq -120(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -128(%rbp)
    movq %rbp, %rax
    movq -128(%rbp), %rcx
    movq %rcx, -8(%rax)
    jmp .L17
.L16:
.L17:
    movq %rbp, %rax
    movq -16(%rax), %rcx
    movq %rcx, -136(%rbp)
    movq $1, %rax
    movq %rax, -144(%rbp)
    movq -136(%rbp), %rax
    movq -144(%rbp), %rcx
    addq %rcx, %rax
    movq %rax, -152(%rbp)
    movq %rbp, %rax
    movq -152(%rbp), %rcx
    movq %rcx, -16(%rax)
    jmp .L9
.L11:
    movq %rbp, %rax
    movq -8(%rax), %rcx
    movq %rcx, -160(%rbp)
    movq -160(%rbp), %rax
    leave
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq $10, %rax
    movq %rax, -8(%rbp)
    leaq print_multiples_of_three(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -8(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $10, %rax
    movq %rax, -16(%rbp)
    leaq sum_even_numbers(%rip), %rax
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
    movq $20, %rax
    movq %rax, -32(%rbp)
    leaq sum_even_numbers(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -32(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq %rax, -40(%rbp)
    leaq __bx_print_int(%rip), %rax
    movq $0, %r10
    pushq %r10
    pushq $0
    movq -40(%rbp), %rdi
    call *%rax
    addq $16, %rsp
    movq $0, %rax
    leave
    ret
