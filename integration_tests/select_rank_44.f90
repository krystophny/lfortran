! Inside a `rank (n)` branch the SELECT RANK selector has rank n (F2018
! 11.1.10.3), so in `rank (0)` it is a scalar and may be passed to a scalar
! dummy argument, as a positional and as a keyword actual argument alike.
program select_rank_44
    implicit none
    real :: s
    real :: a(2)
    real :: b(2, 2)
    real :: total
    integer :: seen_rank

    s = 2.5
    a = [1.0, 3.0]
    b = 1.0

    seen_rank = -1
    call collect(s, total)
    if (seen_rank /= 0) error stop
    if (abs(total - 3.5) > 1.0e-6) error stop

    call collect(a, total)
    if (seen_rank /= 1) error stop
    if (abs(total - 4.0) > 1.0e-6) error stop

    call collect(b, total)
    if (seen_rank /= 2) error stop
    if (abs(total - 4.0) > 1.0e-6) error stop

    print *, seen_rank, total

contains

    ! A non-elemental procedure, so that the rank of each actual argument is
    ! checked against the dummy it corresponds to.
    subroutine take(value, extra, acc)
        real, intent(in) :: value
        real, intent(in), optional :: extra
        real, intent(out) :: acc
        if (present(extra)) then
            acc = value + extra
        else
            acc = value
        end if
    end subroutine take

    subroutine take1(values, acc)
        real, intent(in) :: values(:)
        real, intent(out) :: acc
        acc = sum(values)
    end subroutine take1

    subroutine take2(values, acc)
        real, intent(in) :: values(:, :)
        real, intent(out) :: acc
        acc = sum(values)
    end subroutine take2

    subroutine collect(x, total)
        real, intent(in) :: x(..)
        real, intent(out) :: total
        real :: kw
        select rank (x)
        rank (0)
            seen_rank = 0
            call take(value = x, acc = kw)
            if (abs(kw - 2.5) > 1.0e-6) error stop
            call take(x, 1.0, total)
            if (abs(total - 3.5) > 1.0e-6) error stop
        rank (1)
            seen_rank = 1
            call take1(values = x, acc = total)
        rank (2)
            seen_rank = 2
            call take2(values = x, acc = total)
        rank default
            error stop
        end select
    end subroutine collect

end program select_rank_44
