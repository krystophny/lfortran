module class_150a_m
implicit none

type :: downloader_t
contains
    procedure, nopass :: get_pkg_data
end type downloader_t

contains

subroutine dispatch(arg, used_mock)
    class(downloader_t), optional, intent(in) :: arg
    logical, intent(out) :: used_mock
    class(downloader_t), allocatable :: downloader

    if (present(arg)) then
        downloader = arg
    else
        allocate(downloader)
    end if

    call downloader%get_pkg_data(used_mock)
end subroutine dispatch

subroutine get_pkg_data(used_mock)
    logical, intent(out) :: used_mock

    used_mock = .false.
end subroutine get_pkg_data

end module class_150a_m
