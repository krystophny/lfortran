module modules_73_base
    implicit none

    type, abstract :: value_t
        character(len=:), allocatable :: key
        integer :: origin = 0
    contains
        procedure(destroy_i), deferred :: destroy
        procedure :: match_key
    end type value_t

    type :: key_t
        character(len=:), allocatable :: key
        integer :: origin = 0
    end type key_t

    abstract interface
        subroutine destroy_i(self)
            import value_t
            class(value_t), intent(inout) :: self
        end subroutine destroy_i
    end interface

contains
    pure function match_key(self, key) result(match)
        class(value_t), intent(in) :: self
        character(len=*), intent(in) :: key
        logical :: match
        if (allocated(self%key)) then
            match = key == self%key
        else
            match = .false.
        end if
    end function match_key
end module modules_73_base
