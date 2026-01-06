subroutine test_complex_param
    ! Test COMPLEX PARAMETER with negative real literal (issue #9399)
    ! A negative real assigned to COMPLEX must convert to (real, 0.0)
    implicit none
    complex :: POS_ONE, NEG_ONE, ZERO
    parameter (POS_ONE = 1.0, NEG_ONE = -1.0, ZERO = 0.0)

    if (real(POS_ONE) /= 1.0) error stop
    if (aimag(POS_ONE) /= 0.0) error stop

    if (real(NEG_ONE) /= -1.0) error stop
    if (aimag(NEG_ONE) /= 0.0) error stop

    if (real(ZERO) /= 0.0) error stop
    if (aimag(ZERO) /= 0.0) error stop

    print *, "PASS: complex parameter with negative real literal"
end subroutine

program parameter_16
    call test_complex_param
end program
