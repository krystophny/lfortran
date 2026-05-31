module modules_75_parse_signature
    implicit none

    type :: token_t
        integer :: kind = 1
    end type token_t

    type :: config_t
        integer :: detail = 0
    end type config_t

    type :: context_t
        character(len=:), allocatable :: filename
        character(len=:), allocatable :: source
    end type context_t

    type :: error_t
        character(len=:), allocatable :: message
    end type error_t

    type, abstract :: map_t
    contains
        procedure(destroy_i), deferred :: destroy
    end type map_t

    abstract interface
        subroutine destroy_i(self)
            import :: map_t
            class(map_t), intent(inout) :: self
        end subroutine destroy_i
    end interface

    type, extends(map_t) :: ordered_map_t
        integer, allocatable :: keys(:)
    contains
        procedure :: destroy => ordered_destroy
    end type ordered_map_t

    type, abstract :: value_t
        character(len=:), allocatable :: key
    contains
        procedure(destroy_value_i), deferred :: destroy_value
    end type value_t

    abstract interface
        subroutine destroy_value_i(self)
            import :: value_t
            class(value_t), intent(inout) :: self
        end subroutine destroy_value_i
    end interface

    type, extends(value_t) :: table_t
        class(map_t), allocatable :: map
    contains
        procedure :: destroy_value => destroy_table
    end type table_t

    type :: parser_t
        type(token_t) :: token
        type(table_t), allocatable :: root
        type(table_t), pointer :: current
        type(context_t) :: context
        type(config_t) :: config
        type(error_t), allocatable :: diagnostic
    end type parser_t

    type, abstract :: abstract_lexer_t
    contains
        procedure(next_i), deferred :: next
        procedure(get_info_i), deferred :: get_info
    end type abstract_lexer_t

    abstract interface
        subroutine next_i(lexer, token)
            import :: abstract_lexer_t, token_t
            class(abstract_lexer_t), intent(inout) :: lexer
            type(token_t), intent(inout) :: token
        end subroutine next_i

        subroutine get_info_i(lexer, meta, output)
            import :: abstract_lexer_t
            class(abstract_lexer_t), intent(in) :: lexer
            character(len=*), intent(in) :: meta
            character(len=:), allocatable, intent(out) :: output
        end subroutine get_info_i
    end interface

    type, extends(abstract_lexer_t) :: lexer_t
    contains
        procedure :: next
        procedure :: get_info
    end type lexer_t

contains

    subroutine ordered_destroy(self)
        class(ordered_map_t), intent(inout) :: self

        if (allocated(self%keys)) deallocate(self%keys)
    end subroutine ordered_destroy

    subroutine destroy_table(self)
        class(table_t), intent(inout) :: self

        if (allocated(self%map)) then
            call self%map%destroy()
            deallocate(self%map)
        end if
    end subroutine destroy_table

    subroutine new_map(self)
        class(map_t), allocatable, intent(out) :: self
        type(ordered_map_t), allocatable :: map

        allocate(map)
        allocate(map%keys(1))
        map%keys = 7
        call move_alloc(map, self)
    end subroutine new_map

    function table_ctor() result(table)
        type(table_t) :: table

        call new_map(table%map)
    end function table_ctor

    subroutine next(lexer, token)
        class(lexer_t), intent(inout) :: lexer
        type(token_t), intent(inout) :: token

        token%kind = -2
    end subroutine next

    subroutine get_info(lexer, meta, output)
        class(lexer_t), intent(in) :: lexer
        character(len=*), intent(in) :: meta
        character(len=:), allocatable, intent(out) :: output

        output = meta
    end subroutine get_info

    subroutine new_parser(parser, config)
        type(parser_t), intent(out), target :: parser
        type(config_t), intent(in), optional :: config

        parser%token = token_t(1)
        parser%root = table_ctor()
        parser%current => parser%root
        parser%config = config_t()
        if (present(config)) parser%config = config
    end subroutine new_parser

    subroutine parse_root(parser, lexer)
        class(parser_t), intent(inout) :: parser
        class(abstract_lexer_t), intent(inout) :: lexer

        do while (parser%token%kind /= -2)
            call lexer%next(parser%token)
        end do
    end subroutine parse_root

    subroutine parse(lexer, table, config, context, error)
        class(abstract_lexer_t), intent(inout) :: lexer
        type(table_t), allocatable, intent(out) :: table
        type(config_t), intent(in), optional :: config
        type(context_t), intent(out), optional :: context
        type(error_t), allocatable, intent(out), optional :: error

        type(parser_t) :: parser

        call new_parser(parser, config)
        call parse_root(parser, lexer)

        if (present(error) .and. allocated(parser%diagnostic)) then
            allocate(error)
            error%message = parser%diagnostic%message
        end if
        if (allocated(parser%diagnostic)) return

        call move_alloc(parser%root, table)

        if (present(context)) then
            context = parser%context
            call lexer%get_info("filename", context%filename)
            call lexer%get_info("source", context%source)
        end if
    end subroutine parse

end module modules_75_parse_signature

program modules_75
    use modules_75_parse_signature, only: error_t, lexer_t, parse, table_t
    implicit none

    type(lexer_t) :: lexer
    type(table_t), allocatable :: table
    type(error_t), allocatable :: error

    call parse(lexer, table, error=error)

    if (allocated(error)) error stop 1
    if (.not. allocated(table)) error stop 2
    if (.not. allocated(table%map)) error stop 3

    print *, "PASSED: modules_75"
end program modules_75
