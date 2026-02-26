module initial_02_mod
    implicit none

    type :: metric
        integer :: payload = -1
    contains
        initial :: from_int, from_real
    end type metric

contains

    function from_int(v) result(res)
        integer, intent(in) :: v
        type(metric) :: res
        res%payload = v
    end function from_int

    function from_real(v) result(res)
        real, intent(in) :: v
        type(metric) :: res
        res%payload = int(v) + 100
    end function from_real

end module initial_02_mod

program initial_02
    use initial_02_mod, only: metric
    implicit none
    type(metric) :: a, b

    a = metric(v=3)
    b = metric(v=2.5)

    if (a%payload /= 3) error stop 1
    if (b%payload /= 102) error stop 2
end program initial_02
