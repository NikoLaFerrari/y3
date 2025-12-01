.globl main
.text
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $128, %rsp  # 16 slots
  movq $20, %r11
  movq %r11, -16(%rbp)
  movq -16(%rbp), %r11
  movq %r11, -8(%rbp)
  movq $0, %r11
  movq %r11, -32(%rbp)
  movq -32(%rbp), %r11
  movq %r11, -24(%rbp)
  movq $0, %r11
  movq %r11, -48(%rbp)
  movq -48(%rbp), %r11
  movq %r11, -40(%rbp)
.L0:
  jmp .L1
.L1:
  movq $0, %r11
  movq %r11, -56(%rbp)
  movq -8(%rbp), %r11
  subq -56(%rbp), %r11
  movq %r11, -64(%rbp)
  movq -64(%rbp), %r11
  cmpq $0, %r11
  je .L3
  jmp .L4
.L3:
  jmp .L2
  jmp .L5
.L4:
.L5:
  movq -8(%rbp), %rdi
  callq bx_print_int
  movq $1, %r11
  movq %r11, -72(%rbp)
  movq -72(%rbp), %r11
  movq %r11, -24(%rbp)
  movq $0, %r11
  movq %r11, -80(%rbp)
  movq -80(%rbp), %r11
  movq %r11, -40(%rbp)
.L6:
  jmp .L7
.L7:
  movq -24(%rbp), %r11
  subq -8(%rbp), %r11
  movq %r11, -88(%rbp)
  movq -88(%rbp), %r11
  cmpq $0, %r11
  jg .L9
  jmp .L10
.L9:
  jmp .L8
  jmp .L11
.L10:
.L11:
  movq -40(%rbp), %r11
  addq -24(%rbp), %r11
  movq %r11, -96(%rbp)
  movq -96(%rbp), %r11
  movq %r11, -40(%rbp)
  movq $1, %r11
  movq %r11, -104(%rbp)
  movq -24(%rbp), %r11
  addq -104(%rbp), %r11
  movq %r11, -112(%rbp)
  movq -112(%rbp), %r11
  movq %r11, -24(%rbp)
  jmp .L6
.L8:
  movq -40(%rbp), %rdi
  callq bx_print_int
  movq $1, %r11
  movq %r11, -120(%rbp)
  movq -8(%rbp), %r11
  subq -120(%rbp), %r11
  movq %r11, -128(%rbp)
  movq -128(%rbp), %r11
  movq %r11, -8(%rbp)
  jmp .L0
.L2:
  movq %rbp, %rsp
  popq %rbp
  movq $0, %rax
  retq
