program class_151
use class_151b_m, only: git_target_t
implicit none

type(git_target_t) :: git
logical :: ok

git%url = "seed"
git%object = "revision"

call git%roundtrip(ok)

if (.not. ok) error stop

end program class_151
