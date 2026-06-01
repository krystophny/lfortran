program class_150
use class_150a_m, only: dispatch
use class_150b_m, only: mock_downloader_t
implicit none

type(mock_downloader_t) :: mock
logical :: used_mock

used_mock = .false.
call dispatch(mock, used_mock)

if (.not. used_mock) error stop

end program class_150
