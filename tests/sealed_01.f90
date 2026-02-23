program sealed_01
    implicit none

    type, sealed :: point_t
        integer :: x, y
    end type point_t

    type(point_t) :: p

    p%x = 10
    p%y = 20

    if (p%x /= 10 .or. p%y /= 20) error stop 1
end program sealed_01
