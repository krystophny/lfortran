module modules_73_map_base
    use modules_73_base, only: value_t, key_t
    implicit none

    type, abstract :: map_structure_t
    contains
        procedure(get_i), deferred :: get
        procedure(push_back_i), deferred :: push_back
        procedure(get_keys_i), deferred :: get_keys
    end type map_structure_t

    abstract interface
        subroutine get_i(self, key, ptr)
            import map_structure_t, value_t
            class(map_structure_t), intent(inout), target :: self
            character(len=*), intent(in) :: key
            class(value_t), pointer, intent(out) :: ptr
        end subroutine get_i

        subroutine push_back_i(self, val)
            import map_structure_t, value_t
            class(map_structure_t), intent(inout), target :: self
            class(value_t), allocatable, intent(inout) :: val
        end subroutine push_back_i

        subroutine get_keys_i(self, list)
            import map_structure_t, key_t
            class(map_structure_t), intent(inout), target :: self
            type(key_t), allocatable, intent(out) :: list(:)
        end subroutine get_keys_i
    end interface
end module modules_73_map_base
