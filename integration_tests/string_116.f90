program string_116
implicit none

character(:), allocatable :: text
integer :: pos

text = "ancestor:parent"
pos = index(text, ":")
text = text(pos + 1:)

if (len(text) /= 6) then
    print *, len(text)
    error stop
end if

if (text /= "parent") then
    print *, "[" // text // "]"
    error stop
end if

text = "same"
text = text(:)

if (text /= "same") then
    print *, "[" // text // "]"
    error stop
end if

end program string_116
