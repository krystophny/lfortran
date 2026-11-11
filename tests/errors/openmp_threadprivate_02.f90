! An OpenMP THREADPRIVATE directive is rejected when the variable list is empty.
module openmp_threadprivate_02
    implicit none
    integer :: x
    !$omp threadprivate()
end module
