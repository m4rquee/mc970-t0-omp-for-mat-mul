
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <omp.h>

// Initialize matrices
void initialize_matrices(float *a, float *b, float *c, float *r,
                         unsigned size, unsigned seed) {
  srand(seed);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      a[i * size + j] = rand() % 10;
      b[i * size + j] = rand() % 10;
      c[i * size + j] = rand() % 10;
      r[i * size + j] = 0.0f;
    }
  }
}

void multiply(float *a, float *b, float *r, unsigned size) {
  float sum;

#pragma omp for simd collapse(2) schedule(static) private(sum) nowait
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      sum = 0.0;
      for (int k = 0; k < size; ++k) {
        sum = sum + a[i * size + k] * b[k * size + j];
      }
      r[i * size + j] = sum;
    }
  }
}

void sum(float *a, float *b, unsigned size) {
#pragma omp for simd collapse(2) schedule(static) nowait
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      a[i * size + j] += b[i * size + j];
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
  float *a, *b, *c, *r;
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
  c = (float *)malloc(sizeof(float) * size * size);
  r = (float *)malloc(sizeof(float) * size * size);

  // initialize_matrices with random data
  initialize_matrices(a, b, c, r, size, seed);

  // Compute R = (A * B) + C
  t = omp_get_wtime();

#pragma omp parallel
  {
    // Check for parallelization on this block

    // r = a * b
    multiply(a, b, r, size);

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
  free(c);
  free(r);

  return EXIT_SUCCESS;
}
