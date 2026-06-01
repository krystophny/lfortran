module allocatable_string_empty_cli_append_01_m
    implicit none

    character(len=:), allocatable :: keywords(:)
    character(len=:), allocatable :: values(:)
    integer, allocatable :: counts(:)
    logical, allocatable :: present_in(:)
    logical, allocatable :: mandatory(:)
    logical :: append_values = .true.

contains

    subroutine run_case()
        character(len=:), allocatable :: prefix
        character(len=:), allocatable :: bindir
        character(len=:), allocatable :: path

        call init_dictionary()
        call cmd_args_to_dictionary()

        prefix = sget('prefix')
        bindir = sget('bindir')
        path = prefix // '/' // bindir

        if (path /= '/tmp/liric-fpm-full-install/bin') then
            print *, 'bad['//path//']', len(prefix), len(bindir), len(path)
            error stop
        end if
    end subroutine

    subroutine init_dictionary()
        keywords = [character(len=6) :: 'prefix', 'flag', 'bindir']
        values = [character(len=5) :: '" "', '" "', '"bin"']
        counts = [3, 3, 5]
        present_in = [.false., .false., .false.]
        mandatory = [.false., .true., .false.]
    end subroutine

    function sget(name) result(out)
        character(len=*), intent(in) :: name
        character(len=:), allocatable :: out
        integer :: place

        call locate_key(name, place)
        out = unquote(values(place)(:counts(place)))
    end function

    subroutine cmd_args_to_dictionary()
        integer :: pointer
        character(len=:), allocatable :: lastkeyword
        integer :: i
        integer :: ilength
        character(len=:), allocatable :: current_argument
        character(len=:), allocatable :: current_argument_padded
        character(len=:), allocatable :: oldvalue
        logical :: next_mandatory

        next_mandatory = .false.
        pointer = 0
        lastkeyword = ' '
        i = 1

        do while (get_next_argument())
            current_argument_padded = current_argument // '   '
            if (.not. next_mandatory .and. &
                    current_argument_padded(1:2) == '--') then
                if (lastkeyword /= '') call ifnull()
                call locate_key(current_argument_padded(3:), pointer)
                lastkeyword = trim(current_argument_padded(3:))
                next_mandatory = mandatory(pointer)
            else if (pointer /= 0) then
                oldvalue = get_value(keywords(pointer)) // ' '
                if (oldvalue(1:1) == '"') then
                    current_argument = quote(current_argument(:ilength))
                end if
                call update(keywords(pointer), current_argument)
                pointer = 0
                lastkeyword = ''
                next_mandatory = .false.
            end if
        end do

    contains

        subroutine ifnull()
            oldvalue = get_value(lastkeyword) // ' '
            if (oldvalue(1:1) == '"') then
                call update(lastkeyword, '" "')
            else
                call update(lastkeyword, ' ')
            end if
        end subroutine

        function get_next_argument() result(ok)
            logical :: ok
            character(len=*), parameter :: argv(4) = [character(len=36) :: &
                '--prefix', '/tmp/liric-fpm-full-install', '--flag', &
                '--backend=liric --realloc-lhs-arrays']

            if (i > size(argv)) then
                ok = .false.
                return
            end if

            ok = .true.
            ilength = len_trim(argv(i))
            current_argument = argv(i)(:ilength)
            i = i + 1
        end function

    end subroutine

    function get_value(key) result(out)
        character(len=*), intent(in) :: key
        character(len=:), allocatable :: out
        integer :: place

        call locate_key(key, place)
        out = values(place)(:counts(place))
    end function

    subroutine update(key, val)
        character(len=*), intent(in) :: key
        character(len=*), intent(in) :: val
        integer :: place
        integer :: iilen
        character(len=:), allocatable :: val_local

        call locate_key(key, place)
        val_local = val

        if (present_in(place)) then
            if (append_values) then
                if (values(place)(1:1) == '"') then
                    val_local = '"' // trim(unquote(values(place))) // &
                        ' ' // trim(unquote(val_local)) // '"'
                else
                    val_local = values(place) // ' ' // val_local
                end if
            end if
        end if

        iilen = len_trim(val_local)
        call replace_c(values, val_local, place)
        call replace_i(counts, iilen, place)
        call replace_l(present_in, .true., place)
    end subroutine

    subroutine locate_key(key, place)
        character(len=*), intent(in) :: key
        integer, intent(out) :: place
        integer :: j

        do j = 1, size(keywords)
            if (trim(key) == trim(keywords(j))) then
                place = j
                return
            end if
        end do
        error stop
    end subroutine

    subroutine replace_c(list, value, place)
        character(len=*), intent(in) :: value
        character(len=:), allocatable :: list(:)
        character(len=:), allocatable :: kludge(:)
        integer, intent(in) :: place
        integer :: ii
        integer :: tlen

        tlen = len_trim(value)
        if (len_trim(value) <= len(list)) then
            list(place) = value
        else
            ii = max(tlen, len(list))
            kludge = [character(len=ii) :: list]
            list = kludge
            list(place) = value
        end if
    end subroutine

    subroutine replace_i(list, value, place)
        integer, allocatable :: list(:)
        integer, intent(in) :: value
        integer, intent(in) :: place

        list(place) = value
    end subroutine

    subroutine replace_l(list, value, place)
        logical, allocatable :: list(:)
        logical, intent(in) :: value
        integer, intent(in) :: place

        list(place) = value
    end subroutine

    function quote(str) result(quoted_str)
        character(len=*), intent(in) :: str
        character(len=:), allocatable :: quoted_str

        quoted_str = '"' // trim(str) // '"'
    end function

    pure function unquote(quoted_str) result(unquoted_str)
        character(len=*), intent(in) :: quoted_str
        character(len=:), allocatable :: unquoted_str
        integer :: i
        integer :: iput
        logical :: inside

        allocate(character(len=len(quoted_str)) :: unquoted_str)
        unquoted_str(:) = ''
        iput = 1
        inside = .false.

        do i = 1, len(quoted_str)
            if (quoted_str(i:i) == '"') then
                inside = .not. inside
            else
                unquoted_str(iput:iput) = quoted_str(i:i)
                iput = iput + 1
            end if
        end do

        unquoted_str = unquoted_str(:iput-1)
    end function

end module

program allocatable_string_empty_cli_append_01
    use allocatable_string_empty_cli_append_01_m, only: run_case
    implicit none

    call run_case()
end program
