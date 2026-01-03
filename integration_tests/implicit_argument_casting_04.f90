! Test CHARACTER array type conversion for implicit interface
! When caller has CHARACTER(n) array with PointerArray physical type
! and callee has CHARACTER(*) array with DescriptorArray physical type,
! the codegen must create proper array descriptor.
program implicit_argument_casting_04
    implicit none
    character(1) :: adumma(1)
    external sub_with_char_array

    adumma(1) = ' '
    call sub_with_char_array(adumma)
    print *, "PASS"
end program

subroutine sub_with_char_array(ei)
    implicit none
    character(*) :: ei(*)
    logical :: lsame
    external lsame

    if (lsame(ei(1), ' ')) then
        ! Expected: EI(1) should be space
    else
        error stop "EI(1) is not space"
    end if
end subroutine

logical function lsame(ca, cb)
    implicit none
    character(1), intent(in) :: ca, cb
    lsame = ca == cb
end function
