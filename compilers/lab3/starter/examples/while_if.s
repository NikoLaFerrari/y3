.globl main
.text
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $208, %rsp  # 26 slots
  movq $0, %r11
  movq %r11, -16(%rbp)
  movq -16(%rbp), %r11
  movq %r11, -8(%rbp)
  movq $0, %r11
  movq %r11, -32(%rbp)
  movq -32(%rbp), %r11
  movq %r11, -24(%rbp)
.L0:
  movq $10, %r11
  movq %r11, -40(%rbp)
  movq -8(%rbp), %r11
  subq -40(%rbp), %r11
  movq %r11, -48(%rbp)
  movq -48(%rbp), %r11
  cmpq $0, %r11
  jl .L3
  jmp .L2
.L3:
  movq $9, %r11
  movq %r11, -56(%rbp)
  movq -8(%rbp), %r11
  subq -56(%rbp), %r11
  movq %r11, -64(%rbp)
  movq -64(%rbp), %r11
  cmpq $0, %r11
  je .L2
  jmp .L1
.L1:
  movq $1, %r11
  movq %r11, -72(%rbp)
  movq -8(%rbp), %r11
  addq -72(%rbp), %r11
  movq %r11, -80(%rbp)
  movq -80(%rbp), %r11
  movq %r11, -8(%rbp)
  movq $2, %r11
  movq %r11, -88(%rbp)
  movq -8(%rbp), %rax
  cqto
  movq -88(%rbp), %rbx
  idivq %rbx
  movq %rdx, -96(%rbp)
  movq $1, %r11
  movq %r11, -104(%rbp)
  movq -96(%rbp), %r11
  subq -104(%rbp), %r11
  movq %r11, -112(%rbp)
  movq -112(%rbp), %r11
  cmpq $0, %r11
  je .L4
  jmp .L5
.L4:
  jmp .L0
  jmp .L6
.L5:
  movq -24(%rbp), %r11
  addq -8(%rbp), %r11
  movq %r11, -120(%rbp)
  movq -120(%rbp), %r11
  movq %r11, -24(%rbp)
  movq -24(%rbp), %rdi
  callq bx_print_int
.L6:
  movq $8, %r11
  movq %r11, -128(%rbp)
  movq -8(%rbp), %r11
  subq -128(%rbp), %r11
  movq %r11, -136(%rbp)
  movq -136(%rbp), %r11
  cmpq $0, %r11
  jge .L7
  jmp .L8
.L7:
  jmp .L2
  jmp .L9
.L8:
.L9:
  jmp .L0
.L2:
  movq $10, %r11
  movq %r11, -144(%rbp)
  movq -24(%rbp), %r11
  subq -144(%rbp), %r11
  movq %r11, -152(%rbp)
  movq -152(%rbp), %r11
  cmpq $0, %r11
  jg .L13
  jmp .L11
.L13:
  movq $20, %r11
  movq %r11, -160(%rbp)
  movq -24(%rbp), %r11
  subq -160(%rbp), %r11
  movq %r11, -168(%rbp)
  movq -168(%rbp), %r11
  cmpq $0, %r11
  jl .L10
  jmp .L11
.L10:
  movq $111, %r11
  movq %r11, -176(%rbp)
  movq -176(%rbp), %rdi
  callq bx_print_int
  jmp .L12
.L11:
  movq $20, %r11
  movq %r11, -184(%rbp)
  movq -24(%rbp), %r11
  subq -184(%rbp), %r11
  movq %r11, -192(%rbp)
  movq -192(%rbp), %r11
  cmpq $0, %r11
  jge .L14
  jmp .L15
.L14:
  movq $222, %r11
  movq %r11, -200(%rbp)
  movq -200(%rbp), %rdi
  callq bx_print_int
  jmp .L16
.L15:
  movq $333, %r11
  movq %r11, -208(%rbp)
  movq -208(%rbp), %rdi
  callq bx_print_int
.L16:
.L12:
  movq %rbp, %rsp
  popq %rbp
  movq $0, %rax
  retq
