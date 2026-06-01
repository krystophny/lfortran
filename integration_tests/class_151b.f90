module class_151b_m
use class_151a_m, only: serializable_t
implicit none

type, extends(serializable_t) :: git_target_t
    integer :: descriptor = 200
    character(:), allocatable :: url
    character(:), allocatable :: object
contains
    procedure :: load
end type git_target_t

contains

subroutine load(self, ok)
    class(git_target_t), intent(inout) :: self
    logical, intent(out) :: ok

    self%descriptor = 201
    self%url = "https://example.invalid/repo"
    self%object = "main"

    ok = self%descriptor == 201 .and. &
        self%url == "https://example.invalid/repo" .and. &
        self%object == "main"
end subroutine load

end module class_151b_m
