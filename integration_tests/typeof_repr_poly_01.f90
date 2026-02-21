program typeof_repr_poly_01
    implicit none
    type :: point
        integer :: x
        integer :: y
    end type
    class(*), allocatable :: u
    character(:), allocatable :: s

    allocate(u, source=42)

    s = typeof(u)
    if (index(s, "integer(4)") /= 1) error stop 1

    s = repr(u)
    if (index(s, "integer(4) :: u =") /= 1) error stop 2
    if (index(s, "= 42") == 0) error stop 3

    deallocate(u)
    allocate(u, source=point(1, 2))

    s = typeof(u)
    if (s /= "point") error stop 4

    s = repr(u)
    if (s /= "point :: u = point(1, 2)") error stop 5

    print *, typeof(u)
    print *, repr(u)
end program typeof_repr_poly_01
