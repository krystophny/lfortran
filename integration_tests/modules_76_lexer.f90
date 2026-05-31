module modules_76_lexer
    implicit none
    private

    public :: run_lexer_check

    integer, parameter :: tfc = selected_char_kind("DEFAULT")

    type :: enum_char_t
        character(1, tfc) :: space = tfc_" "
        character(1, tfc) :: hash = tfc_"#"
        character(1, tfc) :: squote = tfc_"'"
        character(3, tfc) :: squote3 = repeat(tfc_"'", 3)
        character(1, tfc) :: dquote = tfc_""""
        character(3, tfc) :: dquote3 = repeat(tfc_"""", 3)
        character(1, tfc) :: backslash = tfc_"\"
        character(1, tfc) :: dot = tfc_"."
        character(1, tfc) :: comma = tfc_","
        character(1, tfc) :: equal = tfc_"="
    end type enum_char_t

    type(enum_char_t), parameter :: char_kind = enum_char_t()

    type :: lexer_t
        character(:, tfc), allocatable :: chunk
    end type lexer_t

contains

    elemental function peek(lexer, pos) result(ch)
        type(lexer_t), intent(in) :: lexer
        integer, intent(in) :: pos
        character(1, tfc) :: ch

        if (pos <= len(lexer%chunk)) then
            ch = lexer%chunk(pos:pos)
        else
            ch = char_kind%space
        end if
    end function peek

    elemental function match(lexer, pos, kind)
        type(lexer_t), intent(in) :: lexer
        integer, intent(in) :: pos
        character(1, tfc), intent(in) :: kind
        logical :: match

        match = peek(lexer, pos) == kind
    end function match

    subroutine run_lexer_check()
        type(lexer_t) :: lexer
        integer :: pos

        allocate(character(3, tfc) :: lexer%chunk)
        lexer%chunk = repeat(tfc_"""", 3)
        pos = 1

        if (.not. all(match(lexer, [pos + 1, pos + 2], &
                char_kind%dquote))) then
            error stop 1
        end if
    end subroutine run_lexer_check

end module modules_76_lexer
