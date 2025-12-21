program do_loop_post_value_01
  use, intrinsic :: iso_fortran_env, only: int32
  implicit none

  integer(int32) :: i, s

  s = 0
  do i = 1, 3
    s = s + i
  end do

  if (s /= 6) error stop
  if (i /= 4) error stop
  print *, "ok"
end program do_loop_post_value_01
