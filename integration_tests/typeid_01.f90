program typeid_01
    implicit none

    type :: point
        integer :: x, y
    end type

    type :: circle
        real(8) :: radius
    end type

    type(type_info) :: tp, tc, tp2

    tp = typeid(point)
    tc = typeid(circle)

    if (trim(type_name(tp)) /= "point") error stop 1
    if (trim(type_name(tc)) /= "circle") error stop 2

    ! typeid returns same handle for same type
    tp2 = typeid(point)
    if (.not. type_same(tp, tp2)) error stop 3

    ! different types produce different handles
    if (type_same(tp, tc)) error stop 4

    ! typeid matches typeof on an instance
    if (.not. type_same(tp, typeof(point(1, 2)))) error stop 5
    if (.not. type_same(tc, typeof(circle(3.14d0)))) error stop 6

    print *, "typeid: all tests passed"
end program typeid_01
