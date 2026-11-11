! An OpenMP THREADPRIVATE directive is rejected when the name is not declared.
module openmp_threadprivate_06
    implicit none
    integer :: x
    !$omp threadprivate(y)
end module
