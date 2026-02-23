module initial_01_mod
    implicit none

    type :: box
        integer :: payload = -1
    contains
        initial :: init
    end type box

contains

    function init(value) result(res)
        integer, intent(in) :: value
        type(box) :: res
        res%payload = value
    end function init

end module initial_01_mod

program initial_01
    use initial_01_mod, only: box
    implicit none
    type(box) :: b

    b = box(value=7)
    if (b%payload /= 7) error stop 1
end program initial_01
