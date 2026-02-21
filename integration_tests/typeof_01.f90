program typeof_01
    implicit none

    real(8) :: x
    integer :: a(3)
    character(:), allocatable :: s

    x = 3.0_8
    a = [1, 2, 3]

    s = typeof(x)
    if (s /= "real(8)") error stop 1

    s = typeof(a)
    if (index(s, "integer(4)") == 0) error stop 2

    print *, typeof(x)
    print *, typeof(a)
end program typeof_01
