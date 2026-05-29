program arrays_func_result_elemental_01
implicit none
real :: x(3), y(3)
integer :: i
x = 3
! Function result sized by size(A) used directly in an elementwise
! expression: the array_op temp holding g(x) must be sized to its real
! extent, not collapsed to a single element (which under-allocates the
! temp so the callee writes out of bounds and trailing elements are lost).
y = g(x) + 1
do i = 1, 3
    if (abs(y(i) - 6.0) > 1.0e-6) error stop
end do
print *, y
contains
    function g(A) result(r)
        real, intent(in) :: A(:)
        real :: r(size(A))
        r = A + 2
    end function
end program
