program typeof_repr_poly_logical_01
    implicit none
    class(*), allocatable :: u
    character(:), allocatable :: s

    allocate(u, source=.true.)

    s = typeof(u)
    if (index(s, "logical(") /= 1) error stop 1

    s = repr(u)
    if (index(s, "logical(") /= 1) error stop 2
    if (index(s, "= T") == 0) error stop 3

    print *, typeof(u)
    print *, repr(u)
end program typeof_repr_poly_logical_01
