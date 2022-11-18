#pragma once

#include "sycl.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ONEMKL_TRANSPOSE_NONTRANS,
    ONEMKL_TRANSPOSE_TRANS,
    ONEMLK_TRANSPOSE_CONJTRANS
} onemklTranspose;

typedef enum {
    ONEMKL_SIDE_LEFT,
    ONEMKL_SIDE_RIGHT
} onemklSide;

typedef enum {
    ONEMKL_UPLO_UPPER,
    ONEMKL_UPLO_LOWER
} onemklUplo;

// XXX: how to expose half in C?
// int onemklHgemm(syclQueue_t device_queue, onemklTranspose transA,
//                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
//                half alpha, const half *A, int64_t lda, const half *B,
//                int64_t ldb, half beta, half *C, int64_t ldc);
int onemklSgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                float alpha, const float *A, int64_t lda, const float *B,
                int64_t ldb, float beta, float *C, int64_t ldc);
int onemklDgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                double alpha, const double *A, int64_t lda, const double *B,
                int64_t ldb, double beta, double *C, int64_t ldc);
int onemklCgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                float _Complex alpha, const float _Complex *A, int64_t lda,
                const float _Complex *B, int64_t ldb, float _Complex beta,
                float _Complex *C, int64_t ldc);
int onemklZgemm(syclQueue_t device_queue, onemklTranspose transA,
                onemklTranspose transB, int64_t m, int64_t n, int64_t k,
                double _Complex alpha, const double _Complex *A, int64_t lda,
                const double _Complex *B, int64_t ldb, double _Complex beta,
                double _Complex *C, int64_t ldc);

void onemklSsymm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, int64_t m, int64_t n,
                float alpha, const float *a, int64_t lda, const float *b,
                int64_t ldb, float beta, float *c, int64_t ldc);

void onemklDsymm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, int64_t m, int64_t n,
                double alpha, const double *a, int64_t lda, const double *b,
                int64_t ldb, double beta, double *c, int64_t ldc);

void onemklCsymm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, int64_t m, int64_t n,
                float _Complex alpha, const float _Complex *a, int64_t lda,
                const float _Complex *b, int64_t ldb, float _Complex beta,
                float _Complex *c, int64_t ldc);

void onemklZsymm(syclQueue_t device_queue, onemklSide left_right,
                onemklUplo upper_lower, int64_t m, int64_t n,
                double _Complex alpha, const double _Complex *a, int64_t lda,
                const double _Complex *b, int64_t ldb, double _Complex beta,
                double _Complex *c, int64_t ldc);

void onemklSsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                onemklTranspose trans, int64_t n, int64_t k, float alpha,
                const float *a, int64_t lda, float beta, float *c, int64_t ldc);

void onemklDsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                onemklTranspose trans, int64_t n, int64_t k, double alpha,
                const double *a, int64_t lda, double beta, double *c, int64_t ldc);

void onemklCsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                onemklTranspose trans, int64_t n, int64_t k, float _Complex alpha,
                const float _Complex *a, int64_t lda, float _Complex beta, float _Complex *c,
                int64_t ldc);

void onemklZsyrk(syclQueue_t device_queue, onemklUplo upper_lower,
                onemklTranspose trans, int64_t n, int64_t k, double _Complex alpha,
                const double _Complex *a, int64_t lda, double _Complex beta, double _Complex *c,
                int64_t ldc);

// Supported Level-1: Nrm2
void onemklDnrm2(syclQueue_t device_queue, int64_t n, const double *x, 
                 int64_t incx, double *result);
void onemklSnrm2(syclQueue_t device_queue, int64_t n, const float *x, 
                 int64_t incx, float *result);
void onemklCnrm2(syclQueue_t device_queue, int64_t n, const float _Complex *x, 
                 int64_t incx, float *result);
void onemklZnrm2(syclQueue_t device_queue, int64_t n, const double _Complex *x, 
                 int64_t incx, double *result);

void onemklDcopy(syclQueue_t device_queue, int64_t n, const double *x,
                 int64_t incx, double *y, int64_t incy);
void onemklScopy(syclQueue_t device_queue, int64_t n, const float *x,
                 int64_t incx, float *y, int64_t incy);
void onemklZcopy(syclQueue_t device_queue, int64_t n, const double _Complex *x,
                 int64_t incx, double _Complex *y, int64_t incy);
void onemklCcopy(syclQueue_t device_queue, int64_t n, const float _Complex *x,
                 int64_t incx, float _Complex *y, int64_t incy);

void onemklDamax(syclQueue_t device_queue, int64_t n, const double *x, int64_t incx,
                 int64_t *result);
void onemklSamax(syclQueue_t device_queue, int64_t n, const float  *x, int64_t incx,
                 int64_t *result);
void onemklZamax(syclQueue_t device_queue, int64_t n, const double _Complex *x, int64_t incx,
                 int64_t *result);
void onemklCamax(syclQueue_t device_queue, int64_t n, const float _Complex *x, int64_t incx,
                 int64_t *result);

void onemklDamin(syclQueue_t device_queue, int64_t n, const double *x, int64_t incx,
                 int64_t *result);
void onemklSamin(syclQueue_t device_queue, int64_t n, const float  *x, int64_t incx,
                 int64_t *result);
void onemklZamin(syclQueue_t device_queue, int64_t n, const double _Complex *x, int64_t incx,
                 int64_t *result);
void onemklCamin(syclQueue_t device_queue, int64_t n, const float _Complex *x, int64_t incx,
                 int64_t *result);

void onemklSswap(syclQueue_t device_queue, int64_t n, float *x, int64_t incx,
                float *y, int64_t incy);
void onemklDswap(syclQueue_t device_queue, int64_t n, double *x, int64_t incx,
                double *y, int64_t incy);
void onemklCswap(syclQueue_t device_queue, int64_t n, float _Complex *x, int64_t incx,
                float _Complex *y, int64_t incy);
void onemklZswap(syclQueue_t device_queue, int64_t n, double _Complex *x, int64_t incx,
                double _Complex *y, int64_t incy);

void onemklDestroy();
#ifdef __cplusplus
}
#endif
