module internal_proc_ptr_host_assoc_01_mod
    implicit none

    abstract interface
        integer function f_iface()
        end function f_iface
    end interface

contains

    subroutine register_like(p)
        procedure(f_iface), pointer, intent(out) :: p
        integer, save :: seed = 41
        p => make_logger
    contains
        integer function make_logger()
            make_logger = seed + 1
        end function make_logger
    end subroutine register_like

end module internal_proc_ptr_host_assoc_01_mod

program internal_proc_ptr_host_assoc_01
    use internal_proc_ptr_host_assoc_01_mod, only: f_iface, register_like
    implicit none

    procedure(f_iface), pointer :: p
    integer :: v

    call register_like(p)
    v = p()
    if (v /= 42) error stop 1
    print *, v
end program internal_proc_ptr_host_assoc_01
