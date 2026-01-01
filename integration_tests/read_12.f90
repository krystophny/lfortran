program read_12
    implicit none
    integer :: u, x
    character(len=*), parameter :: fname = "read_12_tmp.txt"

    open(newunit=u, file=fname, status="replace", action="write")
    write(u, *) 1
    close(u)

    open(newunit=u, file=fname, status="old", action="read")
    read(u, *, end=10) x
    if (x /= 1) then
        print *, "FAIL"
        stop 1
    end if

    read(u, *, end=10) x
    print *, "FAIL"
    stop 2

10  continue
    print *, "PASS"
    close(u, status="delete")
end program read_12
