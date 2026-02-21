program typeof_01
    implicit none

    real(8) :: x
    integer :: a(3)
    type(type_info) :: tx, ta

    x = 3.0_8
    a = [1, 2, 3]

    tx = typeof(x)
    ta = typeof(a)
end program typeof_01
