program intrinsics_503
! execute_command_line passes the command and its length to the C runtime; on
! wasm32 a 32-bit length mismatches the runtime signature and the call traps.
  implicit none
  integer :: exitstat, cmdstat
  character(len=120) :: cmdmsg

  cmdmsg = ""
  exitstat = -1
  cmdstat = -1
  call execute_command_line("true", exitstat=exitstat, cmdstat=cmdstat, cmdmsg=cmdmsg)
  if (cmdstat /= 0) then
    print *, "cmdstat", cmdstat, trim(cmdmsg)
    error stop
  end if
  if (exitstat /= 0) then
    print *, "exitstat", exitstat
    error stop
  end if
  if (len_trim(cmdmsg) /= 0) then
    print *, "cmdmsg", trim(cmdmsg)
    error stop
  end if

  print *, "OK"
end program
