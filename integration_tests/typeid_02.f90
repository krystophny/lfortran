module typeid_02_types
    implicit none
    type :: widget_t
        integer :: id
    end type
end module

module typeid_02_container
    use typeid_02_types, only: widget_t
    implicit none
contains
    subroutine check_typeid()
        type(type_info) :: ti
        ti = typeid(widget_t)
        if (trim(type_name(ti)) /= "widget_t") error stop 1
    end subroutine
end module

program typeid_02
    use typeid_02_container
    use typeid_02_types, only: widget_t
    implicit none
    type(type_info) :: ti
    call check_typeid()
    ti = typeid(widget_t)
    if (trim(type_name(ti)) /= "widget_t") error stop 2
    print *, "PASS"
end program
