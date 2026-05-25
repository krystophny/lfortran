module sealed_02
    implicit none

    type, sealed :: parent_t
        integer :: x
    end type parent_t

    type, extends(parent_t) :: child_t
        integer :: y
    end type child_t
end module sealed_02
