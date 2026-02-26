module internal_proc_arg_01_mod
    implicit none

    abstract interface
        integer function f_iface()
        end function f_iface
    end interface

contains

    subroutine consume(proc, v)
        procedure(f_iface) :: proc
        integer, intent(out) :: v
        v = proc()
    end subroutine consume

    subroutine outer(v)
        integer, intent(out) :: v
        integer :: n
        n = 41
        call consume(local, v)
    contains
        integer function local()
            local = n + 1
        end function local
    end subroutine outer

end module internal_proc_arg_01_mod

program internal_proc_arg_01
    use internal_proc_arg_01_mod, only: outer
    implicit none

    integer :: v
    call outer(v)
    if (v /= 42) error stop 1
    print *, v
end program internal_proc_arg_01
