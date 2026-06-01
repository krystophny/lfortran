module class_tbp_vtable_collision_01_mod
implicit none

logical :: destroyed = .false.
logical :: wrong_call = .false.

type :: base_t
contains
    procedure :: destroy => base_destroy
end type

type, extends(base_t) :: child_t
contains
    procedure :: has_key => child_has_key
end type

contains

subroutine base_destroy(self)
    class(base_t), intent(inout) :: self
    destroyed = .true.
end subroutine

subroutine child_has_key(self)
    class(child_t), intent(inout) :: self
    wrong_call = .true.
end subroutine

end module

program class_tbp_vtable_collision_01
use class_tbp_vtable_collision_01_mod, only: base_t, child_t, destroyed, &
    wrong_call
implicit none

class(base_t), allocatable :: value

allocate(child_t :: value)
call value%destroy()

if (.not. destroyed) error stop
if (wrong_call) error stop

print *, "pass"
end program
