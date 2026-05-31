module modules_73_map
    use modules_73_base, only: value_t, key_t
    use modules_73_map_base, only: map_structure_t
    use modules_73_node, only: node_t
    implicit none

    type, extends(map_structure_t) :: ordered_map_t
        integer :: n = 0
        type(node_t), allocatable :: lst(:)
    contains
        procedure :: get
        procedure :: push_back
        procedure :: get_keys
    end type ordered_map_t

contains
    subroutine new_ordered_map(self)
        type(ordered_map_t), intent(out) :: self
        self%n = 0
        allocate(self%lst(16))
    end subroutine new_ordered_map

    subroutine get(self, key, ptr)
        class(ordered_map_t), intent(inout), target :: self
        character(len=*), intent(in) :: key
        class(value_t), pointer, intent(out) :: ptr
        integer :: i
        nullify(ptr)
        do i = 1, self%n
            if (allocated(self%lst(i)%val)) then
                if (self%lst(i)%val%match_key(key)) then
                    ptr => self%lst(i)%val
                    exit
                end if
            end if
        end do
    end subroutine get

    subroutine push_back(self, val)
        class(ordered_map_t), intent(inout), target :: self
        class(value_t), allocatable, intent(inout) :: val
        self%n = self%n + 1
        call move_alloc(val, self%lst(self%n)%val)
    end subroutine push_back

    subroutine get_keys(self, list)
        class(ordered_map_t), intent(inout), target :: self
        type(key_t), allocatable, intent(out) :: list(:)
        integer :: i
        allocate(list(self%n))
        do i = 1, self%n
            if (allocated(self%lst(i)%val)) then
                if (allocated(self%lst(i)%val%key)) then
                    list(i)%key = self%lst(i)%val%key
                    list(i)%origin = self%lst(i)%val%origin
                end if
            end if
        end do
    end subroutine get_keys
end module modules_73_map
