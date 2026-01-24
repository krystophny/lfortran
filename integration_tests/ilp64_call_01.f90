! Test that scalar integer kind mismatch works with implicit argument casting
! when using -fdefault-integer-8 --implicit-argument-casting
program ilp64_call_01
    implicit none
    integer :: n
    n = 10
    call test_sub(n)
contains
    subroutine test_sub(x)
        integer(4) :: x
        if (x /= 10) error stop
    end subroutine
end program
