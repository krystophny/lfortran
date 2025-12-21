program implied_do_01
    implicit none

    integer :: ii
    integer, allocatable :: vals(:)

    ii = 3
    allocate(vals(9))

    vals = [(ii, ii = 1, 9)]

    if (ii /= 3) error stop 1
    if (vals(ii) /= 3) error stop 2
end program implied_do_01
