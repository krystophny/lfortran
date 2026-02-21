program typeof_01
    implicit none

    real(8) :: x
    integer :: i
    integer :: a(3)
    type(type_info) :: tx, ti, ta

    x = 3.0_8
    i = 10
    a = [1, 2, 3]

    tx = typeof(x)
    ti = typeof(i)
    ta = typeof(a)

    if (trim(type_name(tx)) /= "real(8)") error stop 1
    if (trim(type_name(ti)) /= "integer(4)") error stop 2
    if (trim(type_name(ta)) /= "integer(4)") error stop 3

    if (type_size(tx) /= 8_8) error stop 4
    if (type_size(ti) /= 4_8) error stop 5

    if (.not. type_same(ti, typeof(123))) error stop 6
    if (type_same(tx, ti)) error stop 7
end program typeof_01
