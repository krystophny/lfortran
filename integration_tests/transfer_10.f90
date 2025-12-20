program transfer_10
  use, intrinsic :: iso_fortran_env, only: int8
  implicit none

  character(len=*), parameter :: s = "abc"
  integer(int8), allocatable :: a(:)

  allocate(a(len(s)))
  a = transfer(s, a, len(s))

  if (any(a /= transfer(s, a, len(s)))) error stop
  print *, "ok"
end program transfer_10
