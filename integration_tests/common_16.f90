! Test for ASR verify error when using substring on COMMON block character variable
! Reduced from LAPACK TESTING/LIN/xerbla.f
program common_16
    implicit none
    character(32) :: srnamt
    common / srnamc / srnamt

    srnamt = "HELLO"
    call check_name()
    print *, "PASS"
end program

subroutine check_name()
    implicit none
    character(32) :: srnamt
    common / srnamc / srnamt

    ! This substring operation triggers ASR verify error:
    ! "Var::m_v `srnamt` cannot point outside of its symbol table"
    print *, srnamt(1:5)
end subroutine
