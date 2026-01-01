program write_err_unsupported_01
    implicit none
    integer :: x

    x = 1
    write (*, err=10) x
10  continue
end program write_err_unsupported_01
