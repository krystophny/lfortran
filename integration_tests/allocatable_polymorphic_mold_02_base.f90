module allocatable_polymorphic_mold_02_base
    implicit none
    private
    public :: base_t, table_t, error_t

    type :: error_t
        character(len=:), allocatable :: message
    end type error_t

    type :: table_t
        integer :: value = 0
    end type table_t

    type, abstract :: base_t
    contains
        procedure(to_table), deferred :: dump_to_table
        procedure, non_overridable, private :: dump_to_unit
        generic :: dump => dump_to_table, dump_to_unit
        procedure(from_table), deferred :: load_from_table
        procedure, non_overridable, private :: load_from_unit
        generic :: load => load_from_table, load_from_unit
        procedure, non_overridable :: roundtrip
    end type base_t

    abstract interface
        subroutine to_table(self, table, error)
            import :: base_t, table_t, error_t
            class(base_t), intent(inout) :: self
            type(table_t), intent(inout) :: table
            type(error_t), allocatable, intent(out) :: error
        end subroutine to_table

        subroutine from_table(self, table, error)
            import :: base_t, table_t, error_t
            class(base_t), intent(inout) :: self
            type(table_t), intent(inout) :: table
            type(error_t), allocatable, intent(out) :: error
        end subroutine from_table
    end interface

contains

    subroutine roundtrip(self, error)
        class(base_t), intent(inout) :: self
        type(error_t), allocatable, intent(out) :: error

        integer :: unit
        class(base_t), allocatable :: copy

        open(newunit=unit, form='formatted', action='readwrite', status='scratch')
        call self%dump(unit, error, json=.false.)
        if (allocated(error)) return

        rewind(unit)
        allocate(copy, mold=self)
        call copy%load(unit, error, json=.false.)
        if (allocated(error)) return
        close(unit)
        deallocate(copy)
    end subroutine roundtrip

    subroutine dump_to_unit(self, unit, error, json)
        class(base_t), intent(inout) :: self
        integer, intent(in) :: unit
        type(error_t), allocatable, intent(out) :: error
        logical, optional, intent(in) :: json

        type(table_t) :: table

        call self%dump(table, error)
        write(unit, *) table%value
    end subroutine dump_to_unit

    subroutine load_from_unit(self, unit, error, json)
        class(base_t), intent(inout) :: self
        integer, intent(in) :: unit
        type(error_t), allocatable, intent(out) :: error
        logical, optional, intent(in) :: json

        type(table_t) :: table
        logical :: is_json

        is_json = .false.
        if (present(json)) is_json = json
        if (is_json) error stop

        read(unit, *) table%value
        call self%load(table, error)
    end subroutine load_from_unit
end module allocatable_polymorphic_mold_02_base
