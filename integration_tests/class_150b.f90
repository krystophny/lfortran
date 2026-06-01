module class_150b_m
use class_150a_m, only: downloader_t
implicit none

type, extends(downloader_t) :: mock_downloader_t
contains
    procedure, nopass :: get_pkg_data => mock_get_pkg_data
end type mock_downloader_t

contains

subroutine mock_get_pkg_data(used_mock)
    logical, intent(out) :: used_mock

    used_mock = .true.
end subroutine mock_get_pkg_data

end module class_150b_m
