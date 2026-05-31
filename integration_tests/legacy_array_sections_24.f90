program legacy_array_sections_24
    implicit none

    real, allocatable :: a(:,:)
    integer :: info

    allocate(a(10, 10))
    a = 1.0

    call check_subarray(10, a(1, 1), 10, info)
    if (info /= 0) error stop
    if (abs(a(1, 1) - 1.1) > 0.01) error stop

    print *, "PASSED"
end program

subroutine check_subarray(m, a, lda, info)
    implicit none

    integer, intent(in) :: m, lda
    real, intent(inout) :: a(lda, *)
    integer, intent(out) :: info

    info = 0
    if (m /= 10) then
        info = -1
        return
    end if
    if (lda /= 10) then
        info = -2
        return
    end if

    a(1, 1) = a(1, 1) + 0.1
end subroutine
