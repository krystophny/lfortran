module intent_out_optional_alloc_twice_typebound_m
    implicit none

    type :: container_t
    contains
        procedure, pass :: configure
    end type container_t

contains

    pure subroutine configure(self, log_units)
        class(container_t), intent(in) :: self
        integer, allocatable, intent(out), optional :: log_units(:)

        if (present(log_units)) then
            allocate(log_units(0))
        end if
    end subroutine configure

end module intent_out_optional_alloc_twice_typebound_m
