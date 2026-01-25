program ilp64_maxloc_01
    implicit none
    integer :: res, i
    real :: work(10)
    integer :: indices(1)
    do i = 1, 10
        work(i) = real(i)
    end do
    res = maxloc(work(3:8), 1)
    if (res /= 6) error stop "Test 1 failed"
    res = minloc(work(3:8), 1)
    if (res /= 1) error stop "Test 2 failed"
    indices = maxloc(work)
    if (indices(1) /= 10) error stop "Test 3 failed"
    print *, "PASS"
end program
