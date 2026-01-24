program ilp64_descriptor_01
    implicit none
    real, target :: arr(10)
    real, pointer :: ptr(:)
    integer :: i
    do i = 1, 10
        arr(i) = real(i)
    end do
    ptr => arr(3:8)
    if (size(ptr) /= 6) error stop "size failed"
    if (ptr(1) /= 3.0) error stop "ptr(1) failed"
    print *, "PASS"
end program
