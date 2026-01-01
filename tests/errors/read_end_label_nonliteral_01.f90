program read_end_label_nonliteral_01
    implicit none
    integer :: x
    integer :: lbl

    lbl = 10
    read (*, end=lbl) x
end program read_end_label_nonliteral_01
