module modules_73_table
    use modules_73_base, only: value_t, key_t
    use modules_73_child, only: keyval_t, new_keyval
    use modules_73_map, only: ordered_map_t, new_ordered_map
    use modules_73_map_base, only: map_structure_t
    implicit none

    type, extends(value_t) :: table_t
        class(map_structure_t), allocatable :: map
    contains
        procedure :: get
        procedure :: get_keys
        procedure :: push_back
        procedure :: destroy
    end type table_t

    interface table_t
        module procedure :: table_func
    end interface table_t

contains
    function table_func() result(self)
        type(table_t) :: self
        call new_table(self)
    end function table_func

    subroutine new_table(self)
        type(table_t), intent(out) :: self
        type(ordered_map_t), allocatable :: map
        allocate(map)
        call new_ordered_map(map)
        call move_alloc(map, self%map)
    end subroutine new_table

    subroutine add_keyval_to_table(table, key, ptr, stat)
        class(table_t), intent(inout) :: table
        character(len=*), intent(in) :: key
        type(keyval_t), pointer, intent(out) :: ptr
        integer, intent(out), optional :: stat
        class(value_t), allocatable :: val
        class(value_t), pointer :: tmp
        integer :: istat

        nullify(ptr)
        call new_keyval_(val)
        val%key = key
        call table%push_back(val, istat)

        if (allocated(val)) then
            call val%destroy
            if (present(stat)) stat = -1
            return
        end if

        if (istat == 0) then
            call table%get(key, tmp)
            if (.not. associated(tmp)) then
                if (present(stat)) stat = -1
                return
            end if
            select type(tmp)
            type is(keyval_t)
                ptr => tmp
            class default
                istat = -1
            end select
        end if

        if (present(stat)) stat = istat
    end subroutine add_keyval_to_table

    subroutine get(self, key, ptr)
        class(table_t), intent(inout) :: self
        character(len=*), intent(in) :: key
        class(value_t), pointer, intent(out) :: ptr
        call self%map%get(key, ptr)
    end subroutine get

    subroutine get_keys(self, list)
        class(table_t), intent(inout) :: self
        type(key_t), allocatable, intent(out) :: list(:)
        call self%map%get_keys(list)
    end subroutine get_keys

    subroutine push_back(self, val, stat)
        class(table_t), intent(inout) :: self
        class(value_t), allocatable, intent(inout) :: val
        integer, intent(out) :: stat
        class(value_t), pointer :: ptr

        if (.not. allocated(val)) then
            stat = -1
            return
        end if

        if (.not. allocated(val%key)) then
            stat = -1
            return
        end if

        call self%get(val%key, ptr)
        if (associated(ptr)) then
            stat = -2
            return
        end if

        call self%map%push_back(val)
        stat = 0
    end subroutine push_back

    subroutine new_keyval_(self)
        class(value_t), allocatable, intent(out) :: self
        type(keyval_t), allocatable :: val
        allocate(val)
        call new_keyval(val)
        call move_alloc(val, self)
    end subroutine new_keyval_

    subroutine destroy(self)
        class(table_t), intent(inout) :: self
        if (allocated(self%key)) deallocate(self%key)
        if (allocated(self%map)) deallocate(self%map)
    end subroutine destroy
end module modules_73_table
