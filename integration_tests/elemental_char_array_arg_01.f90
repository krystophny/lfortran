module elemental_char_array_arg_01_mod
implicit none

type :: lexer_t
    character(len=:), allocatable :: chunk
end type

contains

elemental function peek(lexer, pos) result(ch)
    type(lexer_t), intent(in) :: lexer
    integer, intent(in) :: pos
    character(1) :: ch

    if (pos <= len(lexer%chunk)) then
        ch = lexer%chunk(pos:pos)
    else
        ch = " "
    end if
end function

pure function valid_date(string) result(valid)
    character(1), intent(in) :: string(:)
    logical :: valid

    valid = .false.
    if (any(string([5, 8]) /= "-")) return
    valid = .true.
end function

end module

program elemental_char_array_arg_01
use elemental_char_array_arg_01_mod, only: lexer_t, peek, valid_date
implicit none
type(lexer_t) :: lexer
integer :: pos, it
integer, parameter :: offset(*) = [(it, it = 0, 10)]
logical :: has_date

lexer%chunk = "done = 123"
pos = 8
has_date = valid_date(peek(lexer, pos + offset(:10)))
if (has_date) error stop
print *, "pass"
end program
