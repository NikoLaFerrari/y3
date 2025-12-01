.globl main
.text
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $224, %rsp  # 28 slots
  movq $833779, %r11
  movq %r11, -16(%rbp)
  movq -16(%rbp), %r11
  movq %r11, -8(%rbp)
  movq $2, %r11
  movq %r11, -24(%rbp)
  movq -8(%rbp), %rax
  cqto
  movq -24(%rbp), %rbx
  idivq %rbx
  movq %rdx, -32(%rbp)
  movq $0, %r11
  movq %r11, -40(%rbp)
  movq -32(%rbp), %r11
  subq -40(%rbp), %r11
  movq %r11, -48(%rbp)
  movq -48(%rbp), %r11
  cmpq $0, %r11
  je .L0
  jmp .L1
.L0:
  movq $2, %r11
  movq %r11, -56(%rbp)
  movq -56(%rbp), %rdi
  callq bx_print_int
  jmp .L2
.L1:
  movq $3, %r11
  movq %r11, -64(%rbp)
  movq -8(%rbp), %rax
  cqto
  movq -64(%rbp), %rbx
  idivq %rbx
  movq %rdx, -72(%rbp)
  movq $0, %r11
  movq %r11, -80(%rbp)
  movq -72(%rbp), %r11
  subq -80(%rbp), %r11
  movq %r11, -88(%rbp)
  movq -88(%rbp), %r11
  cmpq $0, %r11
  je .L3
  jmp .L4
.L3:
  movq $3, %r11
  movq %r11, -96(%rbp)
  movq -96(%rbp), %rdi
  callq bx_print_int
  jmp .L5
.L4:
  movq $5, %r11
  movq %r11, -104(%rbp)
  movq -8(%rbp), %rax
  cqto
  movq -104(%rbp), %rbx
  idivq %rbx
  movq %rdx, -112(%rbp)
  movq $0, %r11
  movq %r11, -120(%rbp)
  movq -112(%rbp), %r11
  subq -120(%rbp), %r11
  movq %r11, -128(%rbp)
  movq -128(%rbp), %r11
  cmpq $0, %r11
  je .L6
  jmp .L7
.L6:
  movq $5, %r11
  movq %r11, -136(%rbp)
  movq -136(%rbp), %rdi
  callq bx_print_int
  jmp .L8
.L7:
  movq $7, %r11
  movq %r11, -144(%rbp)
  movq -8(%rbp), %rax
  cqto
  movq -144(%rbp), %rbx
  idivq %rbx
  movq %rdx, -152(%rbp)
  movq $0, %r11
  movq %r11, -160(%rbp)
  movq -152(%rbp), %r11
  subq -160(%rbp), %r11
  movq %r11, -168(%rbp)
  movq -168(%rbp), %r11
  cmpq $0, %r11
  je .L9
  jmp .L10
.L9:
  movq $7, %r11
  movq %r11, -176(%rbp)
  movq -176(%rbp), %rdi
  callq bx_print_int
  jmp .L11
.L10:
  movq $11, %r11
  movq %r11, -184(%rbp)
  movq -8(%rbp), %rax
  cqto
  movq -184(%rbp), %rbx
  idivq %rbx
  movq %rdx, -192(%rbp)
  movq $0, %r11
  movq %r11, -200(%rbp)
  movq -192(%rbp), %r11
  subq -200(%rbp), %r11
  movq %r11, -208(%rbp)
  movq -208(%rbp), %r11
  cmpq $0, %r11
  je .L12
  jmp .L13
.L12:
  movq $11, %r11
  movq %r11, -216(%rbp)
  movq -216(%rbp), %rdi
  callq bx_print_int
  jmp .L14
.L13:
.L14:
.L11:
.L8:
.L5:
.L2:
  movq %rbp, %rsp
  popq %rbp
  movq $0, %rax
  retq
