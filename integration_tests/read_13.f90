program read_13
    implicit none
    integer :: u, x
    character(len=*), parameter :: fname = "read_13_tmp.txt"

    open(newunit=u, file=fname, status="replace", action="write")
    write(u, *) 1
    close(u)

    open(newunit=u, file=fname, status="old", action="read")
    read(u, *, end=10) x
    if (x /= 1) then
        print *, "FAIL"
        error stop
    end if

    read(u, *, end=10) x
    print *, "FAIL"
    error stop

10  continue
    print *, "PASS"
    close(u, status="delete")
end program read_13
