program file_open_16
    ! An OPEN of an existing file that may only be read must not ask for
    ! write permission: ACTION='READ' opens it read-only, and with no ACTION
    ! given the processor-dependent mode falls back to read-only, as
    ! GFortran does. A connection that writes still fails.
    implicit none
    integer :: u, ios
    character(len=20) :: line
    character(len=*), parameter :: filename = 'file_open_16_readonly.txt'

    open(newunit=u, file=filename, status='replace')
    write(u, '(a)') 'read-only'
    close(u)
    call execute_command_line('chmod 444 ' // filename)

    open(newunit=u, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) error stop 1
    read(u, '(a)') line
    if (trim(line) /= 'read-only') error stop 2
    close(u)

    open(newunit=u, file=filename, status='old', iostat=ios)
    if (ios /= 0) error stop 3
    read(u, '(a)') line
    if (trim(line) /= 'read-only') error stop 4
    close(u)

    open(newunit=u, file=filename, status='old', access='stream', &
        form='unformatted', iostat=ios)
    if (ios /= 0) error stop 5
    close(u)

    open(newunit=u, file=filename, status='unknown', iostat=ios)
    if (ios /= 0) error stop 6
    close(u)

    open(newunit=u, file=filename, status='old', action='write', iostat=ios)
    if (ios == 0) then
        close(u)
        error stop 7
    end if

    call execute_command_line('chmod u+w ' // filename)
    call execute_command_line('rm -f ' // filename)
    print *, 'exit ok'
end program file_open_16
