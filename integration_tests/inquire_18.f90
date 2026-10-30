module inquire_18_mod
  implicit none
  type :: res_t
    logical :: passed = .false.
  end type res_t
contains
  subroutine check_file(path, r)
    ! The INQUIRE target is a component, which must be passed by address.
    character(*), intent(in) :: path
    type(res_t), intent(out) :: r
    inquire (file=path, exist=r%passed)
  end subroutine check_file
end module inquire_18_mod

program inquire_18
  ! INQUIRE writes through its output specifiers, so each of them must accept
  ! any definable variable, including a derived-type component and an array
  ! element (F2018 12.10.2.10).
  use inquire_18_mod, only: check_file, res_t
  implicit none
  type :: state_t
    logical :: opened = .false.
    integer :: size = -1
    integer :: iostat = -1
  end type state_t
  type(res_t) :: r
  type(state_t) :: st
  logical :: flags(2)
  character(*), parameter :: file = "inquire_18_data.txt"
  integer :: u

  open(newunit=u, file=file, status="replace", action="write", iostat=st%iostat)
  if (st%iostat /= 0) error stop 1
  write(u, '(a)') "hello, world"
  close(u)

  ! Written through a component of an intent(out) dummy.
  call check_file(file, r)
  if (.not. r%passed) error stop 2

  ! Written through an array element.
  flags = .false.
  inquire (file=file, exist=flags(1))
  if (.not. flags(1)) error stop 3
  inquire (file=file // ".absent", exist=flags(2))
  if (flags(2)) error stop 4

  ! Several output specifiers at once.
  open(newunit=u, file=file, status="old", action="read", iostat=st%iostat)
  if (st%iostat /= 0) error stop 5
  inquire (unit=u, opened=st%opened, size=st%size, iostat=st%iostat)
  if (st%iostat /= 0) error stop 6
  if (.not. st%opened) error stop 7
  if (st%size <= 0) error stop 8
  close(u, status="delete")

  print *, "ok"
end program inquire_18
