program typeof_repr_member_01
    implicit none

    type :: base
        integer :: x
    end type base

    type, extends(base) :: child
        integer :: y
    end type child

    type(child) :: c
    type(type_info) :: tb, tc, tp
    character(len=:), allocatable :: s

    tb = typeof(base(1))
    tc = typeof(c)

    if (tb%name() /= type_name(tb)) error stop 1
    if (tc%name() /= type_name(tc)) error stop 2

    if (tb%size() /= type_size(tb)) error stop 3

    tp = tc%parent()
    if (tp%name() /= type_name(type_parent(tc))) error stop 4

    if (.not. tc%same(typeof(c))) error stop 5
    if (tc%same(tb)) error stop 6

    if (.not. tc%extends(tb)) error stop 7
    if (tb%extends(tc)) error stop 8

    s = repr(tc)
    if (s /= tc%name()) error stop 9

    print *, tc%name()
end program typeof_repr_member_01
