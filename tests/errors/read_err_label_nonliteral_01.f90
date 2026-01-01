program read_err_label_nonliteral_01
    implicit none
    integer :: x
    integer :: lbl

    lbl = 10
    read (*, err=lbl) x
end program read_err_label_nonliteral_01
