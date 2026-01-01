program write_end_unsupported_01
    implicit none
    integer :: x

    x = 1
    write (*, end=10) x
10  continue
end program write_end_unsupported_01
