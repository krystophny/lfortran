module pass_array_by_data_13_mod
  implicit none
contains
  subroutine host(a)
    integer, intent(inout) :: a(:)
    call inner(a, 1)
  contains
    recursive subroutine inner(a, i)
      integer, intent(inout) :: a(:)
      integer, intent(in) :: i
      if (i < size(a)) call inner(a, i + 1)
    end subroutine inner
  end subroutine host
end module pass_array_by_data_13_mod

program pass_array_by_data_13
  use pass_array_by_data_13_mod, only: host
  implicit none
  integer :: a(3)
  a = [1, 2, 3]
  call host(a)
  if (any(a /= [1, 2, 3])) error stop
end program pass_array_by_data_13
