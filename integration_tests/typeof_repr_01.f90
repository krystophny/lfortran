program typeof_repr_01
    implicit none

    real(8) :: x
    integer :: a(3)
    type :: point
        integer :: x
        integer :: y
    end type
    type(point) :: p
    character(:), allocatable :: s
    type(type_info) :: t

    x = 3.0_8
    a = [1, 2, 3]
    p = point(1, 2)

    t = typeof(x)
    if (repr(t) /= "real(8)") error stop 1
    if (type_name(t) /= "real(8)") error stop 5
    if (type_size(t) /= 8_8) error stop 6
    if (.not. type_same(t, typeof(x))) error stop 7
    if (type_name(type_parent(t)) /= "~null_type") error stop 8

    s = repr(x)
    if (index(s, "real(8) :: x =") /= 1) error stop 2

    s = repr(a)
    if (s /= "integer(4) :: a(3) = [1, 2, 3]") error stop 3

    s = repr(p)
    if (s /= "point :: p = point(1, 2)") error stop 4

    print *, t
    print *, repr(x)
end program typeof_repr_01
