program read_err_label_nonliteral_02
    implicit none
    integer :: x

    read (*, err=10.0) x
end program read_err_label_nonliteral_02
