    .text
    .globl main
main:
  pushq %rbp
  movq %rsp, %rbp
  subq $72, %rsp
.LB0:
  movq $0, -8(%rbp)
  movq $0, -16(%rbp)
.LL0:
  movq $10, -32(%rbp)
  movq -8(%rbp), %r11
  cmpq -32(%rbp), %r11
  setl %r10b
  movzbq %r10b, %r11
  movq %r11, -24(%rbp)
  movq -24(%rbp), %r11
  cmpq $0, %r11
  jne .LL1
  jmp .LL2
.LL1:
  movq $5, -48(%rbp)
  movq -8(%rbp), %r11
  cmpq -48(%rbp), %r11
  sete %r10b
  movzbq %r10b, %r11
  movq %r11, -40(%rbp)
  movq -40(%rbp), %r11
  cmpq $0, %r11
  jne .LL3
  jmp .LL5
.LL3:
  jmp .LL2
.LL5:
  movq -16(%rbp), %r11
  addq -8(%rbp), %r11
  movq %r11, -56(%rbp)
  movq -56(%rbp), %r11
  movq %r11, -16(%rbp)
  movq $1, -64(%rbp)
  movq -8(%rbp), %r11
  addq -64(%rbp), %r11
  movq %r11, -72(%rbp)
  movq -72(%rbp), %r11
  movq %r11, -8(%rbp)
  jmp .LL0
.LL2:
  movq -16(%rbp), %rdi
  call _bx_print_int
  xorq %rax, %rax
  movq %rbp, %rsp
  popq %rbp
  retq
