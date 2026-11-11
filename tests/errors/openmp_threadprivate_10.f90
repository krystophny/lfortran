! An OpenMP THREADPRIVATE directive is rejected when the variable is in a common block.
module openmp_threadprivate_10
    implicit none
    integer :: x
    common /otp_blk/ x
    !$omp threadprivate(x)
end module
