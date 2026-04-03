module initial_03_mod
    implicit none

    type :: item
        integer :: value = -1
    contains
        initial :: init
    end type item

    abstract interface
        function item_ctor(v) result(res)
            import :: item
            integer, intent(in) :: v
            type(item) :: res
        end function item_ctor
    end interface

contains

    function init(v) result(res)
        integer, intent(in) :: v
        type(item) :: res
        res%value = v
    end function init

end module initial_03_mod

program initial_03
    use initial_03_mod, only: item, item_ctor
    implicit none
    procedure(item_ctor), pointer :: ctor
    type(item) :: a, b

    ctor => item%init
    a = ctor(11)
    b = item%init(13)

    if (a%value /= 11) error stop 1
    if (b%value /= 13) error stop 2
end program initial_03
