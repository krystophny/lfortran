program read_13
    implicit none

    integer, parameter :: int64 = selected_int_kind(18)
    integer(int64) :: x
    integer :: unit, ios

    open(newunit=unit, status="scratch", action="readwrite")
    write(unit, "(A)") "abc"
    rewind(unit)
    read(unit, *, iostat=ios) x
    close(unit)

    if (ios <= 0) error stop
end program read_13
