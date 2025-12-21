program do_loop_post_value_02
    use, intrinsic :: iso_fortran_env, only: int64, error_unit
    implicit none

    integer(int64) :: buf(0:23)
    integer(int64) :: bend, remainder, step, c
    integer(int64) :: i

    do i = 0_int64, 23_int64
        buf(i) = i
    end do

    bend = 0_int64
    remainder = 16_int64

    c = 0_int64
    do step = 0_int64, bend - 1_int64, 4_int64
        c = c + buf(step)
    end do

    ! Standard Fortran semantics: if the DO loop executes 0 iterations,
    ! the loop variable becomes the initial value.
    if (step /= 0_int64) then
        write(error_unit, '(a,i0)') 'step expected 0, got ', step
        error stop
    end if

    if (remainder >= 16_int64) then
        c = c + buf(step)
    end if

    if (c /= 0_int64) then
        write(error_unit, '(a,i0)') 'c expected 0, got ', c
        error stop
    end if

    write(*, '(a)') 'OK'
end program do_loop_post_value_02
