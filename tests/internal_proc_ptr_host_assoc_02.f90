program internal_proc_ptr_host_assoc_02
    implicit none

    abstract interface
        integer function f_iface()
        end function f_iface
    end interface

    procedure(f_iface), pointer :: p
    integer :: n, v

    n = 40
    p => local
    n = 41

    v = p()
    if (v /= 42) error stop 1

    call consume_proc(p, v)
    if (v /= 42) error stop 2

    print *, v

contains

    integer function local()
        local = n + 1
    end function local

    subroutine consume_proc(proc, out)
        procedure(f_iface) :: proc
        integer, intent(out) :: out
        out = proc()
    end subroutine consume_proc

end program internal_proc_ptr_host_assoc_02
