module pointer_18_mod
implicit none

type, abstract :: generic_value
end type

type, extends(generic_value) :: boolean_value
    logical :: raw
end type

type :: keyval_t
    class(generic_value), allocatable :: val
contains
    procedure :: set_boolean
    procedure :: get_boolean
end type

contains

subroutine set_boolean(self, val)
    class(keyval_t), intent(inout) :: self
    logical, intent(in) :: val
    type(boolean_value), allocatable :: tmp
    allocate(tmp)
    tmp%raw = val
    call move_alloc(tmp, self%val)
end subroutine

subroutine get_boolean(self, val)
    class(keyval_t), intent(in) :: self
    logical, pointer, intent(out) :: val
    select type (ptr => self%val)
    type is (boolean_value)
        val => ptr%raw
    class default
        nullify(val)
    end select
end subroutine

end module

program pointer_18
use pointer_18_mod, only: keyval_t
implicit none

type(keyval_t) :: keyval
logical, pointer :: ptr

call keyval%set_boolean(.false.)
call keyval%get_boolean(ptr)
if (.not. associated(ptr)) error stop
if (ptr) error stop

call keyval%set_boolean(.true.)
call keyval%get_boolean(ptr)
if (.not. ptr) error stop

print *, "pass"
end program
