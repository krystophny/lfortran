program repr_01
    implicit none

    integer :: i
    real(8) :: x
    character(:), allocatable :: s

    i = 42
    x = 3.0_8

    s = repr(i)
    if (index(s, "integer(4) :: i =") /= 1) error stop 1

    s = repr(x)
    if (index(s, "real(8) :: x =") /= 1) error stop 2

    s = repr(i + 1)
    if (index(s, "integer(4) :: it =") /= 1) error stop 3

    print *, repr(i)
    print *, repr(x)
    print *, repr(i + 1)
end program repr_01
