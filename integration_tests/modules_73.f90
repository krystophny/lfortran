program modules_73
    use modules_73_base, only: key_t
    use modules_73_child, only: keyval_t
    use modules_73_table, only: table_t, add_keyval_to_table
    implicit none

    type(table_t) :: table
    type(key_t), allocatable :: keys(:)
    type(keyval_t), pointer :: ptr
    integer :: stat

    table = table_t()
    call add_keyval_to_table(table, "name", ptr, stat)
    if (stat /= 0) error stop 1
    if (.not. associated(ptr)) error stop 2

    call table%get_keys(keys)
    if (size(keys) /= 1) error stop 3
    if (.not. allocated(keys(1)%key)) error stop 4
    if (len(keys(1)%key) /= 4) error stop 5
    if (keys(1)%key /= "name") error stop 6

    print *, "PASSED: modules_73"
end program modules_73
