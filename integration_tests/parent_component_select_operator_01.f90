module parent_component_select_operator_01_mod
implicit none

type, abstract :: serializable_t
end type

type, extends(serializable_t) :: config_t
    integer :: value = 1
contains
    procedure :: is_same => config_same
    generic :: operator(==) => is_same
end type

type, extends(config_t) :: node_t
    integer :: extra = 2
end type

contains

logical function config_same(this, that)
    class(config_t), intent(in) :: this
    type(config_t), intent(in) :: that
    config_same = this%value == that%value
end function

logical function node_same(this, that)
    type(node_t), intent(in) :: this
    class(serializable_t), intent(in) :: that
    node_same = .false.
    select type (other => that)
    type is (node_t)
        if (.not. (this%config_t == other%config_t)) return
        if (this%extra /= other%extra) return
    class default
        return
    end select
    node_same = .true.
end function

end module

program parent_component_select_operator_01
use parent_component_select_operator_01_mod, only: serializable_t, node_t, &
    node_same
implicit none
type(node_t) :: a
class(serializable_t), allocatable :: b

allocate(node_t :: b)
select type (b)
type is (node_t)
    b%value = a%value
    b%extra = a%extra
end select

if (.not. node_same(a, b)) error stop
print *, "pass"
end program
