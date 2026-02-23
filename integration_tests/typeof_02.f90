module typeof_02_mod
    implicit none

    type :: entry_t
        type(type_info) :: key
    end type

    type :: holder_t
        type(entry_t) :: entries(4)
        integer :: n = 0
    end type

contains

    subroutine check_name(ti, expected)
        type(type_info), intent(in) :: ti
        character(*), intent(in) :: expected
        if (trim(type_name(ti)) /= expected) error stop 1
    end subroutine

    subroutine check_size(ti, expected)
        type(type_info), intent(in) :: ti
        integer(8), intent(in) :: expected
        if (type_size(ti) /= expected) error stop 2
    end subroutine

    subroutine add_entry(h, key)
        type(holder_t), intent(inout) :: h
        type(type_info), intent(in) :: key
        h%n = h%n + 1
        h%entries(h%n)%key = key
    end subroutine

    subroutine verify_entry(h, idx, expected_name)
        type(holder_t), intent(in) :: h
        integer, intent(in) :: idx
        character(*), intent(in) :: expected_name
        if (trim(type_name(h%entries(idx)%key)) /= expected_name) error stop 3
    end subroutine

end module

program typeof_02
    use typeof_02_mod
    implicit none

    type(type_info) :: ti
    type(holder_t) :: h
    real(8) :: x
    integer :: i

    x = 1.0d0
    i = 42

    ! Test 1: type_info passed as subroutine parameter
    ti = typeof(x)
    call check_name(ti, "real(8)")
    call check_size(ti, 8_8)

    ti = typeof(i)
    call check_name(ti, "integer(4)")
    call check_size(ti, 4_8)

    ! Test 2: type_info stored in struct, then read back
    call add_entry(h, typeof(x))
    call add_entry(h, typeof(i))
    call verify_entry(h, 1, "real(8)")
    call verify_entry(h, 2, "integer(4)")

end program typeof_02
