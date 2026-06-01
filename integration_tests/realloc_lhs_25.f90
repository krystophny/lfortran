program realloc_lhs_25
    implicit none

    type :: settings_t
        character(len=:), allocatable :: name(:)
    end type

    type(settings_t) :: settings
    character(len=15), allocatable :: actual(:)

    settings%name = [character(len=8) :: "proj1", "p2", "project3"]

    allocate(character(len=15) :: actual(0))
    actual = settings%name

    if (len(actual) /= 15) error stop 1
    if (size(actual) /= 3) error stop 2
    call check(actual(1), "proj1")
    call check(actual(2), "p2")
    call check(actual(3), "project3")

    print *, "PASS"

contains

    subroutine check(value, expected)
        character(len=*), intent(in) :: value
        character(len=*), intent(in) :: expected
        integer :: i

        if (value(1:len(expected)) /= expected) error stop 3
        do i = len(expected) + 1, len(value)
            if (iachar(value(i:i)) /= iachar(" ")) error stop 4
        end do
    end subroutine

end program
