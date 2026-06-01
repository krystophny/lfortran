program file_73
implicit none

character(len=:), allocatable :: text
character(len=64) :: line
integer :: i, iostat, nlines, unit

allocate(character(len=0) :: text)
do i = 1, 220
    text = text // new_line('a') // 'name = "dependency-name"'
end do

open(newunit=unit, form='formatted', status='scratch', action='readwrite', &
    recl=1073741824)
call write_text(unit, text)
rewind(unit)

nlines = 0
do
    read(unit, '(a)', iostat=iostat) line
    if (iostat /= 0) exit
    nlines = nlines + 1
    if (trim(line) == 'n') error stop
end do
close(unit)

if (nlines /= 221) then
    print *, nlines
    error stop
end if

contains

subroutine write_text(unit, text)
    integer, intent(in) :: unit
    character(*), intent(in) :: text

    write(unit, *) text
end subroutine

end program file_73
