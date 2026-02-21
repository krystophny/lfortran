program typeof_repr_member_01
    implicit none

    type :: base
        integer :: x
    end type base

    type, extends(base) :: child
        integer :: y
    end type child

    type(child) :: c
    integer :: a(3)
    class(*), allocatable :: u
    type(type_info) :: tb, tc, tp
    character(len=:), allocatable :: s

    a = [1, 2, 3]
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

    ! Intrinsic array: member/procedural reflection APIs must match.
    tp = typeof(a)
    if (tp%name() /= type_name(tp)) error stop 10
    if (tp%size() /= type_size(tp)) error stop 11
    s = repr(a)
    if (s /= "integer(4) :: a(3) = [1, 2, 3]") error stop 12

    ! Polymorphic value: both APIs must still agree on reflected type.
    allocate(u, source=42)
    tp = typeof(u)
    if (tp%name() /= type_name(tp)) error stop 13
    if (.not. tp%same(typeof(42))) error stop 14
    s = repr(u)
    if (index(s, "integer(4) :: u =") /= 1) error stop 15

    print *, tc%name()
end program typeof_repr_member_01
