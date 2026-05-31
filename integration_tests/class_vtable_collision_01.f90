module class_vtable_collision_01_m
    implicit none

    type, abstract :: base_t
    contains
        procedure(base_same), deferred :: same
    end type

    abstract interface
        logical function base_same(self, that)
            import :: base_t
            class(base_t), intent(in) :: self
            class(base_t), intent(in) :: that
        end function
    end interface

    type :: dependency_t
        integer :: unit = 77
    end type

    type, extends(base_t) :: tree_t
    contains
        procedure :: same => tree_same
        generic :: add => add_dependency
        procedure :: add_project
        procedure :: add_dependency
        generic :: load_cache => load_cache_from_file, load_cache_from_unit
        procedure :: load_cache_from_file
        procedure :: load_cache_from_unit
    end type

contains

    logical function tree_same(self, that)
        class(tree_t), intent(in) :: self
        class(base_t), intent(in) :: that

        tree_same = .true.
    end function

    subroutine add_project(self, error)
        class(tree_t), intent(inout) :: self
        type(dependency_t) :: dependency
        integer, intent(out) :: error

        call self%add(dependency, error)
    end subroutine

    subroutine add_dependency(self, dependency, error)
        class(tree_t), intent(inout) :: self
        type(dependency_t), intent(in) :: dependency
        integer, intent(out) :: error

        error = 11
    end subroutine

    subroutine load_cache_from_file(self, file, error)
        class(tree_t), intent(inout) :: self
        character(len=*), intent(in) :: file
        integer, intent(out) :: error

        error = -1
    end subroutine

    subroutine load_cache_from_unit(self, unit, error)
        class(tree_t), intent(inout) :: self
        integer, intent(in) :: unit
        integer, intent(out) :: error

        error = unit
    end subroutine

end module

program class_vtable_collision_01
    use class_vtable_collision_01_m, only: tree_t
    implicit none

    type(tree_t) :: tree
    integer :: error

    call tree%add_project(error)
    if (error /= 11) error stop
end program
