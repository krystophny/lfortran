module allocatable_polymorphic_mold_02_child
    use allocatable_polymorphic_mold_02_base, only: base_t, error_t, table_t
    implicit none
    private
    public :: child_t, child_revision

    type, extends(base_t) :: child_t
        integer :: value = 0
        character(len=:), allocatable :: url
        character(len=:), allocatable :: object
    contains
        procedure :: dump_to_table => child_dump_to_table
        procedure :: load_from_table => child_load_from_table
    end type child_t

contains

    function child_revision(url, object) result(self)
        character(len=*), intent(in) :: url
        character(len=*), intent(in) :: object
        type(child_t) :: self

        self%value = 42
        self%url = url
        self%object = object
    end function child_revision

    subroutine child_dump_to_table(self, table, error)
        class(child_t), intent(inout) :: self
        type(table_t), intent(inout) :: table
        type(error_t), allocatable, intent(out) :: error

        table%value = self%value
    end subroutine child_dump_to_table

    subroutine child_load_from_table(self, table, error)
        class(child_t), intent(inout) :: self
        type(table_t), intent(inout) :: table
        type(error_t), allocatable, intent(out) :: error

        self%value = table%value
    end subroutine child_load_from_table
end module allocatable_polymorphic_mold_02_child
