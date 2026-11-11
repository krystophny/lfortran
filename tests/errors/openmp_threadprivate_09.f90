! An OpenMP THREADPRIVATE directive is rejected when the variable is local to a subroutine.
subroutine openmp_threadprivate_09()
    implicit none
    integer, save :: x
    !$omp threadprivate(x)
    x = 1
end subroutine
