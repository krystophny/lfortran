module modules_73_child
    use modules_73_base, only: value_t
    implicit none

    type, abstract :: generic_t
    end type generic_t

    type, extends(value_t) :: keyval_t
        class(generic_t), allocatable :: val
        integer :: origin_value = 0
    contains
        procedure :: destroy
    end type keyval_t

contains
    subroutine new_keyval(self)
        type(keyval_t), intent(out) :: self
        associate(self => self); end associate
    end subroutine new_keyval

    subroutine destroy(self)
        class(keyval_t), intent(inout) :: self
        if (allocated(self%key)) deallocate(self%key)
        if (allocated(self%val)) deallocate(self%val)
    end subroutine destroy
end module modules_73_child
