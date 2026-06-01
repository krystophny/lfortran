module class_151a_m
implicit none

type, abstract :: serializable_t
contains
    procedure(load_i), deferred :: load
    procedure, non_overridable :: roundtrip
end type serializable_t

abstract interface
    subroutine load_i(self, ok)
        import serializable_t
        class(serializable_t), intent(inout) :: self
        logical, intent(out) :: ok
    end subroutine load_i
end interface

contains

subroutine roundtrip(self, ok)
    class(serializable_t), intent(inout) :: self
    logical, intent(out) :: ok
    class(serializable_t), allocatable :: copy

    allocate(copy, mold=self)
    call copy%load(ok)
end subroutine roundtrip

end module class_151a_m
