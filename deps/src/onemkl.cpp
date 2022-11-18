#include "onemkl.h"
#include "sycl.hpp"

#include <oneapi/mkl.hpp>

// gemm

// https://spec.oneapi.io/versions/1.0-rev-1/elements/oneMKL/source/domains/blas/gemm.html

oneapi::mkl::transpose convert(onemklTranspose val) {
    switch (val) {
    case ONEMKL_TRANSPOSE_NONTRANS:
        return oneapi::mkl::transpose::nontrans;
    case ONEMKL_TRANSPOSE_TRANS:
        return oneapi::mkl::transpose::trans;
    case ONEMLK_TRANSPOSE_CONJTRANS:
        return oneapi::mkl::transpose::conjtrans;
    }
}

oneapi::mkl::side convert(onemklSide val) {
    switch (val) {
    case ONEMKL_SIDE_LEFT:
        return oneapi::mkl::side::left;
    case ONEMKL_SIDE_RIGHT:
        return oneapi::mkl::side::right;
    }
}

oneapi::mkl::uplo convert(onemklUplo val) {
    switch(val) {
    case ONEMKL_UPLO_UPPER:
        return oneapi::mkl::uplo::upper;
    case ONEMKL_UPLO_LOWER:
        return oneapi::mkl::uplo::lower;
    }
}

extern "C" int onemklHgemm(syclQueue_t device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, sycl::half alpha, const sycl::half *A, int64_t lda,
                           const sycl::half *B, int64_t ldb, sycl::half beta, sycl::half *C,
                           int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue->val, convert(transA),
                                          convert(transB), m, n, k, alpha, A,
                                          lda, B, ldb, beta, C, ldc);
    return 0;
}

extern "C" int onemklSgemm(syclQueue_t device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, float alpha, const float *A, int64_t lda,
                           const float *B, int64_t ldb, float beta, float *C,
                           int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue->val, convert(transA),
                                          convert(transB), m, n, k, alpha, A,
                                          lda, B, ldb, beta, C, ldc);
    return 0;
}

extern "C" int onemklDgemm(syclQueue_t device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, double alpha, const double *A,
                           int64_t lda, const double *B, int64_t ldb,
                           double beta, double *C, int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(device_queue->val, convert(transA),
                                          convert(transB), m, n, k, alpha, A,
                                          lda, B, ldb, beta, C, ldc);
    return 0;
}

extern "C" int onemklCgemm(syclQueue_t device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, float _Complex alpha,
                           const float _Complex *A, int64_t lda,
                           const float _Complex *B, int64_t ldb,
                           float _Complex beta, float _Complex *C,
                           int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(
        device_queue->val, convert(transA), convert(transB), m, n, k, alpha,
        reinterpret_cast<const std::complex<float> *>(A), lda,
        reinterpret_cast<const std::complex<float> *>(B), ldb, beta,
        reinterpret_cast<std::complex<float> *>(C), ldc);
    return 0;
}

extern "C" int onemklZgemm(syclQueue_t device_queue, onemklTranspose transA,
                           onemklTranspose transB, int64_t m, int64_t n,
                           int64_t k, double _Complex alpha,
                           const double _Complex *A, int64_t lda,
                           const double _Complex *B, int64_t ldb,
                           double _Complex beta, double _Complex *C,
                           int64_t ldc) {
    oneapi::mkl::blas::column_major::gemm(
        device_queue->val, convert(transA), convert(transB), m, n, k, alpha,
        reinterpret_cast<const std::complex<double> *>(A), lda,
        reinterpret_cast<const std::complex<double> *>(B), ldb, beta,
        reinterpret_cast<std::complex<double> *>(C), ldc);
    return 0;
}

extern "C" void onemklSsymm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            float alpha, const float *a, int64_t lda, const float *b,
                            int64_t ldb, float beta, float *c, int64_t ldc) {
    oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right),
                                          convert(upper_lower), m, n, alpha, a, lda, b,
                                          ldb, beta, c, ldc);
}

extern "C" void onemklDsymm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            double alpha, const double *a, int64_t lda, const double *b,
                            int64_t ldb, double beta, double *c, int64_t ldc) {
    oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right),
                                          convert(upper_lower), m, n, alpha, a, lda, b,
                                          ldb, beta, c, ldc);
}

extern "C" void onemklCsymm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            float _Complex alpha, const float _Complex *a, int64_t lda,
                            const float _Complex *b, int64_t ldb, float _Complex beta,
                            float _Complex *c, int64_t ldc) {
    oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right),
                                          convert(upper_lower), m, n,
                                          static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a),
                                          lda, reinterpret_cast<const std::complex<float> *>(b),
                                          ldb, beta, reinterpret_cast<std::complex<float> *>(c), ldc);
}

extern "C" void onemklZsymm(syclQueue_t device_queue, onemklSide left_right,
                            onemklUplo upper_lower, int64_t m, int64_t n,
                            double _Complex alpha, const double _Complex *a, int64_t lda,
                            const double _Complex *b, int64_t ldb, double _Complex beta,
                            double _Complex *c, int64_t ldc) {
    oneapi::mkl::blas::column_major::symm(device_queue->val, convert(left_right),
                                          convert(upper_lower), m, n,
                                          static_cast<std::complex<double> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a), lda,
                                          reinterpret_cast<const std::complex<double> *>(b), ldb,
                                          static_cast<std::complex<double> >(beta),
                                          reinterpret_cast<std::complex<double> *>(c), ldc);
}

extern "C" void onemklSsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                            onemklTranspose trans, int64_t n, int64_t k, float alpha,
                            const float *a, int64_t lda, float beta, float *c, int64_t ldc) {
    oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower), convert(trans),
                                          n, k, alpha, a, lda, beta, c, ldc);
}

extern "C" void onemklDsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                            onemklTranspose trans, int64_t n, int64_t k, double alpha,
                            const double *a, int64_t lda, double beta, double *c, int64_t ldc) {
    oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower), convert(trans),
                                          n, k, alpha, a, lda, beta, c, ldc);
}

extern "C" void onemklCsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                            onemklTranspose trans, int64_t n, int64_t k, float _Complex alpha,
                            const float _Complex *a, int64_t lda, float _Complex beta, float _Complex *c,
                            int64_t ldc) {
    oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower), convert(trans),
                                          n, k, static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<float> *>(a), lda,
                                          static_cast<std::complex<float> >(beta),
                                          reinterpret_cast<std::complex<float> *>(c), ldc);
}

extern "C" void onemklZsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                            onemklTranspose trans, int64_t n, int64_t k, double _Complex alpha,
                            const double _Complex *a, int64_t lda, double _Complex beta, double _Complex *c,
                            int64_t ldc) {
    oneapi::mkl::blas::column_major::syrk(device_queue->val, convert(upper_lower), convert(trans),
                                          n, k, static_cast<std::complex<float> >(alpha),
                                          reinterpret_cast<const std::complex<double> *>(a), lda,
                                          static_cast<std::complex<double> >(beta),
                                          reinterpret_cast<std::complex<double> *>(c), ldc);
}

extern "C" void onemklDnrm2(syclQueue_t device_queue, int64_t n, const double *x, 
                            int64_t incx, double *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n, x, incx, result);
    status.wait();
}

extern "C" void onemklSnrm2(syclQueue_t device_queue, int64_t n, const float *x, 
                            int64_t incx, float *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n, x, incx, result);
    status.wait();
}

extern "C" void onemklCnrm2(syclQueue_t device_queue, int64_t n, const float _Complex *x, 
                            int64_t incx, float *result) {   
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n, 
                    reinterpret_cast<const std::complex<float> *>(x), incx, result);
    status.wait();
}

extern "C" void onemklZnrm2(syclQueue_t device_queue, int64_t n, const double _Complex *x, 
                            int64_t incx, double *result) {
    auto status = oneapi::mkl::blas::column_major::nrm2(device_queue->val, n, 
                    reinterpret_cast<const std::complex<double> *>(x), incx, result);
    status.wait();
}

extern "C" void onemklDcopy(syclQueue_t device_queue, int64_t n, const double *x,
                            int64_t incx, double *y, int64_t incy) {
    oneapi::mkl::blas::column_major::copy(device_queue->val, n, x, incx, y, incy);
}

extern "C" void onemklScopy(syclQueue_t device_queue, int64_t n, const float *x,
                            int64_t incx, float *y, int64_t incy) {
    oneapi::mkl::blas::column_major::copy(device_queue->val, n, x, incx, y, incy);
}

extern "C" void onemklZcopy(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, double _Complex *y, int64_t incy) {
    oneapi::mkl::blas::column_major::copy(device_queue->val, n,
        reinterpret_cast<const std::complex<double> *>(x), incx,
        reinterpret_cast<std::complex<double> *>(y), incy);
}

extern "C" void onemklCcopy(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, float _Complex *y, int64_t incy) {
    oneapi::mkl::blas::column_major::copy(device_queue->val, n, 
        reinterpret_cast<const std::complex<float> *>(x), incx, 
        reinterpret_cast<std::complex<float> *>(y), incy);
}

extern "C" void onemklDamax(syclQueue_t device_queue, int64_t n, const double *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n, x, incx, result);
    status.wait();
}
extern "C" void onemklSamax(syclQueue_t device_queue, int64_t n, const float  *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n, x, incx, result);
    status.wait();
}
extern "C" void onemklZamax(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n,
                            reinterpret_cast<const std::complex<double> *>(x), incx, result);
    status.wait();
}
extern "C" void onemklCamax(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamax(device_queue->val, n,
                            reinterpret_cast<const std::complex<float> *>(x), incx, result);
    status.wait();
}

extern "C" void onemklDamin(syclQueue_t device_queue, int64_t n, const double *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n, x, incx, result);
    status.wait();
}
extern "C" void onemklSamin(syclQueue_t device_queue, int64_t n, const float  *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n, x, incx, result);
    status.wait();
}
extern "C" void onemklZamin(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n,
                            reinterpret_cast<const std::complex<double> *>(x), incx, result);
    status.wait();
}
extern "C" void onemklCamin(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                            int64_t incx, int64_t *result){
    auto status = oneapi::mkl::blas::column_major::iamin(device_queue->val, n,
                            reinterpret_cast<const std::complex<float> *>(x), incx, result);
    status.wait();
}

extern "C" void onemklSswap(syclQueue_t device_queue, int64_t n, float *x, int64_t incx,\
                            float *y, int64_t incy){
    oneapi::mkl::blas::column_major::swap(device_queue->val, n, x, incx, y, incy);
}

extern "C" void onemklDswap(syclQueue_t device_queue, int64_t n, double *x, int64_t incx,
                            double *y, int64_t incy){
    oneapi::mkl::blas::column_major::swap(device_queue->val, n, x, incx, y, incy);
}

extern "C" void onemklCswap(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx,
                            float _Complex *y, int64_t incy){
    oneapi::mkl::blas::column_major::swap(device_queue->val, n,
                            reinterpret_cast<std::complex<float> *>(x), incx,
                            reinterpret_cast<std::complex<float> *>(y), incy);
}

extern "C" void onemklZswap(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx,
                            double _Complex *y, int64_t incy){
    oneapi::mkl::blas::column_major::swap(device_queue->val, n,
                            reinterpret_cast<std::complex<double> *>(x), incx,
                            reinterpret_cast<std::complex<double> *>(y), incy);
}

// other

// oneMKL keeps a cache of SYCL queues and tries to destroy them when unloading the library.
// that is incompatible with oneAPI.jl destroying queues before that, so expose a function
// to manually wipe the device cache when we're destroying queues.

namespace oneapi {
namespace mkl {
namespace gpu {
int clean_gpu_caches();
}
}
}

extern "C" void onemklDestroy() {
    oneapi::mkl::gpu::clean_gpu_caches();
}
