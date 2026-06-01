module pointer_19_mod
implicit none

contains

subroutine copy_i8(value, result)
    integer(8), intent(in) :: value
    integer(8), intent(out) :: result

    result = value
end subroutine

subroutine get_i8_pointer(ptr)
    integer(8), pointer, intent(out) :: ptr
    integer(8), target, save :: stored = 42_8

    ptr => stored
end subroutine

function identity_i8(value) result(result)
    integer(8), intent(in) :: value
    integer(8) :: result

    result = value
end function

end module

program pointer_19
use pointer_19_mod, only: copy_i8, get_i8_pointer, identity_i8
implicit none

integer(8), target :: value
integer(8), pointer :: direct_ptr
integer(8), pointer :: dummy_ptr
integer(8) :: result

value = 42_8
direct_ptr => value

call copy_i8(direct_ptr, result)
if (result /= 42_8) error stop 1

if (identity_i8(direct_ptr) /= 42_8) error stop 2

call get_i8_pointer(dummy_ptr)

call copy_i8(dummy_ptr, result)
if (result /= 42_8) error stop 3

if (identity_i8(dummy_ptr) /= 42_8) error stop 4
end program
