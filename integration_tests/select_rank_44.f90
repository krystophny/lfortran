! The rank that a SELECT RANK selector takes inside the construct lasts only
! for the construct (F2018 11.1.10.3). After `end select` the dummy is assumed
! rank again, in the same procedure and in any later procedure that has a
! same-named dummy.
module select_rank_44_m
    implicit none
contains

    subroutine rank_after_end_select(x)
        integer, intent(in) :: x(..)
        select rank (x)
        rank (1)
            error stop "unexpected rank 1"
        end select
        if (rank(x) /= 2) error stop "rank leaked past end select"
        select rank (x)
        rank (2)
            if (size(x) /= 6) error stop "wrong size in rank 2 block"
        end select
        if (rank(x) /= 2) error stop "rank leaked past the second end select"
    end subroutine rank_after_end_select

    ! Dummy named as in rank_after_end_select, called after that construct.
    subroutine rank_in_later_procedure(x)
        integer, intent(in) :: x(..)
        if (rank(x) /= 3) error stop "rank leaked into a later procedure"
    end subroutine rank_in_later_procedure

    subroutine present_after_select(x)
        real, intent(in), optional :: x(..)
        if (.not. present(x)) error stop "x shall be present"
    end subroutine present_after_select

    subroutine present_of_absent(x)
        real, intent(in), optional :: x(..)
        if (present(x)) error stop "x shall be absent"
    end subroutine present_of_absent
end module select_rank_44_m

program select_rank_44
    use select_rank_44_m
    implicit none
    integer :: a2(2, 3), a3(2, 1, 2)
    real :: r1(4)

    a2 = 7
    a3 = 1
    call rank_after_end_select(a2)
    call rank_in_later_procedure(a3)
    call present_after_select(r1)
    call present_of_absent()
    print *, "PASS"
end program select_rank_44
