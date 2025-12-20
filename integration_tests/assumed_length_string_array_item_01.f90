program assumed_length_string_array_item_01
    implicit none

    character(len=3) :: a(2)

    a(1) = "bbb"
    a(2) = "aaa"
    call swap_if_needed(a)

    if (a(1) /= "aaa") error stop 1
    if (a(2) /= "bbb") error stop 2

contains

    subroutine swap_if_needed(a)
        character(len=*), intent(inout) :: a(:)
        character(len=len(a(1))) :: tmp

        if (a(1) > a(2)) then
            tmp = a(1)
            a(1) = a(2)
            a(2) = tmp
        end if
    end subroutine swap_if_needed
end program assumed_length_string_array_item_01
