program realloc_lhs_char_array_constructor_01
    use iso_c_binding, only: c_null_char
    implicit none

    character(len=:), allocatable :: list(:)
    character(len=:), allocatable :: value
    integer :: nul

    list = [character(len=2) :: ' ']
    value = '--backend=liric --realloc-lhs-arrays'
    call replace_c(list, value, 1)

    nul = index(list(1), c_null_char)
    if (nul /= 0) error stop
    if (len(list) /= len(value)) error stop
    if (trim(list(1)) /= value) error stop

contains

    subroutine replace_c(list, value, place)
        character(len=:), allocatable, intent(inout) :: list(:)
        character(len=*), intent(in) :: value
        integer, intent(in) :: place
        character(len=:), allocatable :: kludge(:)
        integer :: ii
        integer :: tlen

        tlen = len_trim(value)
        if (len_trim(value) <= len(list)) then
            list(place) = value
        else
            ii = max(tlen, len(list))
            kludge = [character(len=ii) :: list]
            list = kludge
            list(place) = value
        end if
    end subroutine

end program
