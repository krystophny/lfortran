program internal_proc_ptr_host_assoc_01
    implicit none

    abstract interface
        integer function f_iface()
        end function f_iface
    end interface

    procedure(f_iface), pointer :: p
    call register_like(p)

contains

    subroutine register_like(ptr)
        procedure(f_iface), pointer, intent(out) :: ptr
        integer :: n
        n = 41
        ptr => local
    contains
        integer function local()
            local = n + 1
        end function local
    end subroutine register_like

end program internal_proc_ptr_host_assoc_01
