program bindc_cchar_array_intent_in_writeback_01
    use iso_c_binding, only: c_char, c_null_char
    implicit none

    character(kind=c_char, len=1), allocatable :: buffer(:)

    interface
        subroutine fill_buffer(buf) bind(c, name="fill_buffer")
            import c_char
            character(kind=c_char, len=1), intent(in) :: buf(*)
        end subroutine
    end interface

    allocate(buffer(4))
    buffer = 'X'

    call fill_buffer(buffer)

    if (buffer(1) /= 'O') error stop
    if (buffer(2) /= 'K') error stop
    if (buffer(3) /= c_null_char) error stop
end program
