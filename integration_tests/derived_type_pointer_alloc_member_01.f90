program derived_type_pointer_alloc_member_01
  use, intrinsic :: iso_fortran_env, only: int32
  implicit none

  type :: t
    integer(int32), allocatable :: a(:)
  end type t

  type(t), pointer :: p

  allocate(p)
  allocate(p%a(3))
  p%a = [1_int32, 2_int32, 3_int32]

  if (any(p%a /= [1_int32, 2_int32, 3_int32])) error stop
  print *, "ok"
end program derived_type_pointer_alloc_member_01
