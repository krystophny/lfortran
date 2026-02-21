program typeof_repr_poly_logical_01
    implicit none
    class(*), allocatable :: u
    character(:), allocatable :: s
    type(type_info) :: t

    allocate(u, source=.true.)

    t = typeof(u)
    s = type_name(t)
    if (index(s, "logical(") /= 1) error stop 1

    s = repr(u)
    if (index(s, "logical(") /= 1) error stop 2
    if (index(s, "= T") == 0) error stop 3

    print *, t
    print *, repr(u)
end program typeof_repr_poly_logical_01
