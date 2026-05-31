module finalization_pointer_scope_01_m
    implicit none

    integer :: final_count = 0

    type :: payload_t
        integer :: value = 0
    contains
        final :: finalize_payload
    end type

    type :: holder_t
        type(payload_t) :: payload
    end type

contains

    subroutine finalize_payload(self)
        type(payload_t), intent(inout) :: self
        self%value = -1
        final_count = final_count + 1
    end subroutine

    subroutine leave_scope()
        type(holder_t), pointer :: p
        nullify(p)
    end subroutine

end module

program finalization_pointer_scope_01
    use finalization_pointer_scope_01_m, only: final_count, leave_scope
    implicit none

    call leave_scope()

    if (final_count /= 0) error stop
end program
