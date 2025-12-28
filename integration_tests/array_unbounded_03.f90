program array_unbounded_03
    implicit none

    call driver(3)

contains

    subroutine driver(n)
        integer, intent(in) :: n
        real :: work(n)
        integer :: i

        do i = 1, n
            work(i) = real(i)
        end do

        call callee(work)
    end subroutine driver

end program array_unbounded_03

subroutine callee(x)
    implicit none
    real :: x(*)

    if (abs((x(1) + x(2) + x(3)) - 6.0) > 1e-6) error stop
end subroutine callee
