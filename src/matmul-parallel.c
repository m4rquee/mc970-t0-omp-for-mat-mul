
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <omp.h>

// Initialize matrices
void initialize_matrices(float *a, float *b, float *bT, float *c, float *r,
                         unsigned size, unsigned seed) {
  srand(seed);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      a[i * size + j] = rand() % 10;
      b[i * size + j] = rand() % 10;
      c[i * size + j] = rand() % 10;
      bT[i * size + j] = 0.0f;
      r[i * size + j] = 0.0f;
    }
  }
}

void multiply(const float * __restrict__ a, const float * __restrict__ bT, float * __restrict__ r, unsigned size) {
  float sum;
  int ioff, joff;

#pragma omp for collapse(2) schedule(static) private(sum, ioff, joff) nowait
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      sum = 0.0;
      ioff = i * size, joff = j * size;
      #pragma omp simd
      for (int k = 0; k < size; k++) {
        sum += a[ioff + k] * bT[joff + k];
      }
      r[ioff + j] = sum;
    }
  }
}

void sum(float * __restrict__ a, const float * __restrict__ b, unsigned size) {
#pragma omp for simd collapse(2) schedule(static) nowait
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      a[i * size + j] += b[i * size + j];
    }
  }
}

void transpose(float * __restrict__ mT, const float * __restrict__ m, unsigned size) {
#pragma omp for simd collapse(2) schedule(static) nowait
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      mT[i * size + j] = m[j * size + i];
    }
  }
}

// Output matrix to stdout
void print_matrix(float *r, unsigned size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      printf(" %5.1f", r[i * size + j]);
    }
    printf("\n");
  }
}

int main(int argc, char *argv[]) {
  float *a, *b, *bT, *c, *r;
  unsigned seed, size;
  double t;
  FILE *input;

  if (argc < 2) {
    fprintf(stderr, "Error: missing path to input file\n");
    return 1;
  }

  if ((input = fopen(argv[1], "r")) == NULL) {
    fprintf(stderr, "Error: could not open file\n");
    return 1;
  }

  // Read inputs
  fscanf(input, "%u", &size);
  fscanf(input, "%u", &seed);

  // Do not change this line
  omp_set_num_threads(4);

  // Allocate matrices
  a = (float *)malloc(sizeof(float) * size * size);
  b = (float *)malloc(sizeof(float) * size * size);
  bT = (float *)malloc(sizeof(float) * size * size);
  c = (float *)malloc(sizeof(float) * size * size);
  r = (float *)malloc(sizeof(float) * size * size);

  // initialize_matrices with random data
  initialize_matrices(a, b, bT, c, r, size, seed);

  // Compute R = (A * B) + C
  t = omp_get_wtime();

#pragma omp parallel if(size >= 100) // if data is large enough
  {
    // Check for parallelization on this block

    // bT = b^T
    transpose(bT, b, size); // to increase data locality

    // r = a * b (b is expected to be transposed)
    multiply(a, bT, r, size);

    // r += c
    sum(r, c, size);
  }

  t = omp_get_wtime() - t;

  // Show result
  print_matrix(r, size);

  // Output elapsed time
  fprintf(stderr, "%lf\n", t);

  // Release memory
  free(a);
  free(b);
  free(bT);
  free(c);
  free(r);

  return EXIT_SUCCESS;
}
