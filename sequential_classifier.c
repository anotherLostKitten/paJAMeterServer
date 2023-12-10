// Imports
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#include "mnist_constants.h"

double calculate_L2_norm(double* vector, int size) {

    // Init the double sum
    double sum_of_squares = 0.0;

    // Sum the squares
    for (int i = 0; i < size; i++) {
        sum_of_squares += vector[i] * vector[i];
    }

    // Calculate the square root
    return sqrt(sum_of_squares);
}

// Function to generate random weight between -0.5 and .5
double generate_random_weight() {
    return (((double) rand()) / ((double) RAND_MAX)) -.5 ;
}


// Dot product function for vectors
double dot_product(double* a, double* b, int size) {
    double result = 0.;

    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }

    return result;
}

void gradient(double* gradient_output, double* x, double* w, int label) {
    // Note n is FEATURE_SIZE_W_BIAS since that is the feature size of our image
    // label is the expected numerical result; we do a softmax on the model outputs
    // x is (D+1)x1 array -- the pixels of the image as an array - D is number of features - +1 because of bias
    // w is (D+1)x10 array -- we automatically assume the transpose
    // y_hat is by definition wT*x = y_hat
    // Since our convention has w as (D+1)x10 we can just use dot product;
    // The bias is a 1x10 value

    double y_hat[10], sum = 0;
    for (int n = 0; n < 10; n++)
        sum += y_hat[n] = exp(dot_product(x, &w[n * FEATURE_SIZE_W_BIAS], FEATURE_SIZE_W_BIAS));
    for (int n = 0; n < 10; n++)
        y_hat[n] = y_hat[n] / sum - (label == n ? 1. : 0.);

    // Calculate the gradient vector -- this is a (D+1)x1 sized vector
    for (int i = 0; i < FEATURE_SIZE_W_BIAS; i++)
        for (int n = 0; n < 10; n++)
            gradient_output[i + FEATURE_SIZE_W_BIAS * n] = x[i] * y_hat[n];
}

// Function used to add a new column to the features matrix - a bias column
void add_bias_column(uint8_t* x, double* x_with_bias) {
    for (int i = 0; i < FEATURE_SIZE_W_BIAS; i++) {
        if (i == 0)
            x_with_bias[i] = 1.;
        else
            x_with_bias[i] = (double)x[i] / 255.;
    }
}

// Initialize weights needed for gradient descent - add an extra row for bias
void init_weights(double* w){
    for (int i = 0; i < FEATURE_SIZE_W_BIAS; i++)
        w[i] = generate_random_weight();
}

#define BATCH_SIZE 1

double gradient_output[FEATURE_SIZE_W_BIAS * 10 * BATCH_SIZE];

// Run Gradient Descent
void run_gradient_descent(double learning_rate, struct image_data* datum, double* w, int batch){

    // Step 1: Add a bias column to the feature vector - ONLY USE x_with_bias from here
    double x[FEATURE_SIZE_W_BIAS];
    add_bias_column(datum->data, x);

    // Calculate the gradient
    gradient(&gradient_output[FEATURE_SIZE_W_BIAS * 10 * batch], x, w, datum->label);

    // Perform gradient descent -- aka update the weights based on the gradient

    if (batch == BATCH_SIZE - 1)
        for (int b = 0; b < BATCH_SIZE; b++)
            for (int i = 0; i < FEATURE_SIZE_W_BIAS * 10; i++)
                w[i] -= learning_rate * gradient_output[FEATURE_SIZE_W_BIAS * 10 * b + i];

    // // Step 3: Calculate the l2 norm in order to get the model error
    // double error = calculate_L2_norm(gradient_output, FEATURE_SIZE_W_BIAS * 10);

    // // Step 4: Print out the error
    // printf("The model error is: %lf\n", error);
}

int main() {
    struct image_data* data = read_data(MNIST_TRAINING_IMAGES, "training_sets/train-images.idx3-ubyte", "training_sets/train-labels.idx1-ubyte");

    double w[FEATURE_SIZE_W_BIAS * 10];
    for (int i=0; i < FEATURE_SIZE_W_BIAS * 10; i++)
        w[i] = generate_random_weight();

    clock_t start = clock();

    int batch = 0;
    for (int i = 0; i < MNIST_TRAINING_IMAGES; i++) {
        run_gradient_descent(LEARNING_RATE, &data[i], w, batch);
        if (BATCH_SIZE > 1)
            batch = (batch + 1) % BATCH_SIZE;
    }

    if (BATCH_SIZE > 1)
        if (batch)
            for (int b = 0; b < batch - 1; b++)
                for (int i = 0; i < FEATURE_SIZE_W_BIAS * 10; i++)
                    w[i] -= LEARNING_RATE * gradient_output[FEATURE_SIZE_W_BIAS * 10 * b + i];

    clock_t end = clock();

    double elapsed = (end - start) / (double)CLOCKS_PER_SEC;

    printf("elapsed: %lf\n", elapsed);

    free(data);

    data = read_data(MNIST_TESTING_IMAGES, "training_sets/t10k-images.idx3-ubyte", "training_sets/t10k-labels.idx1-ubyte");

    int success = 0;

    for (int img = 0; img < MNIST_TESTING_IMAGES; img++) {
        double maxc = 0;
        int maxn = -1;
        for (int n = 0; n < 10; n++) {
            double conf = w[785 * n + 784];
            for (int i = 0; i < 784; i++)
                conf += w[785 * n + i] * (double)data[img].data[i] / 255.0;
            if (conf > maxc) {
                maxc = conf;
                maxn = n;
            }
        }
        if (maxn == data[img].label)
            success++;
    }

    free(data);

    printf("accuracy: %d / %d (%lf%%)\n", success, MNIST_TESTING_IMAGES, (double)success / (double)MNIST_TESTING_IMAGES * 100.0);
}
