! An OpenMP THREADPRIVATE directive is rejected when the name is a derived type, not a variable.
module openmp_threadprivate_07
    implicit none
    type :: t
        integer :: x
    end type
    !$omp threadprivate(t)
end module
