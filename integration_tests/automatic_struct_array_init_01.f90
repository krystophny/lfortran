module automatic_struct_array_init_01_m
    use iso_c_binding, only: c_f_pointer, c_int64_t, c_ptr, c_size_t
    implicit none

    type :: string_t
        character(len=:), allocatable :: s
    end type

contains

    subroutine poison(n)
        integer, intent(in) :: n
        type(c_ptr) :: mem
        integer(c_int64_t), pointer :: words(:)

        interface
            function c_malloc(size) bind(c, name="malloc") result(ptr)
                import c_ptr, c_size_t
                integer(c_size_t), value :: size
                type(c_ptr) :: ptr
            end function

            subroutine c_free(ptr) bind(c, name="free")
                import c_ptr
                type(c_ptr), value :: ptr
            end subroutine
        end interface

        mem = c_malloc(int(n * 16, c_size_t))
        call c_f_pointer(mem, words, [2 * n])
        words = 12345_c_int64_t
        call c_free(mem)
    end subroutine

    subroutine fill(n)
        integer, intent(in) :: n
        type(string_t) :: dirs_temp(n)
        integer :: i

        do i = 1, n
            dirs_temp(i)%s = ' '
        end do
        dirs_temp(1)%s = 'app'
        if (dirs_temp(1)%s /= 'app') error stop
    end subroutine

end module

program automatic_struct_array_init_01
    use automatic_struct_array_init_01_m, only: fill, poison
    implicit none

    call poison(3)
    call fill(3)
    print *, 'pass'
end program
