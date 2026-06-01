module associate_54_mod
    implicit none

    type :: lexer_t
        integer :: input_position = 0
        integer :: input_length = -1
    contains
        procedure :: init
    end type

contains

    subroutine init(self, n)
        class(lexer_t), intent(inout) :: self
        integer, intent(in) :: n

        self%input_position = 0
        self%input_length = n
    end subroutine
end module

program associate_54
    use associate_54_mod, only: lexer_t
    implicit none

    type(lexer_t) :: lex

    associate(pos => lex%input_position, length => lex%input_length)
        call lex%init(3)
        if (pos /= 0) error stop
        if (length /= 3) error stop
        do while (pos < length)
            pos = pos + 1
        end do
    end associate

    if (lex%input_position /= 3) error stop
end program
