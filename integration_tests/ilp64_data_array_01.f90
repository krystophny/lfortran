program ilp64_data_array_01
    implicit none
    integer :: ipivot(4, 4)
    integer :: i, j
    data ipivot /1, 2, 3, 4, 2, 1, 4, 3, 3, 4, 1, 2, 4, 3, 2, 1/
    
    ! Verify first column
    if (ipivot(1, 1) /= 1) error stop "Test 1 failed"
    if (ipivot(2, 1) /= 2) error stop "Test 2 failed"
    if (ipivot(3, 1) /= 3) error stop "Test 3 failed"
    if (ipivot(4, 1) /= 4) error stop "Test 4 failed"
    
    ! Verify second column
    if (ipivot(1, 2) /= 2) error stop "Test 5 failed"
    if (ipivot(2, 2) /= 1) error stop "Test 6 failed"
    
    ! Verify last element
    if (ipivot(4, 4) /= 1) error stop "Test 7 failed"
    
    print *, "PASS"
end program
