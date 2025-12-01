.globl main
.text
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $112, %rsp  # 14 slots
  movq $20, %r11
  movq %r11, -16(%rbp)
  movq -16(%rbp), %r11
  movq %r11, -8(%rbp)
  movq $0, %r11
  movq %r11, -32(%rbp)
  movq -32(%rbp), %r11
  movq %r11, -24(%rbp)
  movq $1, %r11
  movq %r11, -48(%rbp)
  movq -48(%rbp), %r11
  movq %r11, -40(%rbp)
  movq $0, %r11
  movq %r11, -64(%rbp)
  movq -64(%rbp), %r11
  movq %r11, -56(%rbp)
.L0:
  movq $0, %r11
  movq %r11, -72(%rbp)
  movq -8(%rbp), %r11
  subq -72(%rbp), %r11
  movq %r11, -80(%rbp)
  movq -80(%rbp), %r11
  cmpq $0, %r11
  jg .L1
  jmp .L2
.L1:
  movq $1, %r11
  movq %r11, -88(%rbp)
  movq -8(%rbp), %r11
  subq -88(%rbp), %r11
  movq %r11, -96(%rbp)
  movq -96(%rbp), %r11
  movq %r11, -8(%rbp)
  movq -24(%rbp), %rdi
  callq bx_print_int
  movq -24(%rbp), %r11
  addq -40(%rbp), %r11
  movq %r11, -104(%rbp)
  movq -104(%rbp), %r11
  movq %r11, -56(%rbp)
  movq -40(%rbp), %r11
  movq %r11, -24(%rbp)
  movq -56(%rbp), %r11
  movq %r11, -40(%rbp)
  jmp .L0
.L2:
  movq %rbp, %rsp
  popq %rbp
  movq $0, %rax
  retq
