program do_loop_04
    integer :: n = 2, k
    do k = 2, n
        print *, k
    end do

    ! Legacy LFortran semantics (default without the --use-loop-variable-after-loop flag)

    if (k /= 2) error stop

    print *, "k after = ", k
end program
