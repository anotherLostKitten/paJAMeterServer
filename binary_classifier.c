// Imports
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#include "mnist_constants.h"

double calculate_L2_norm(double *vector, int size) {

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


// Logistic Function
double logistic(double z) {
    return 1. / (1. + exp(-z));
}

// Dot product function for vectors
double dot_product(double *a, double *b, int size) {
    double result = 0.;

    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }

    return result;
}

void gradient(double *gradient_output, double *x, double *w, double y){
    // Note n is FEATURE_SIZE_W_BIAS since that is the feature size of our image
    // y is 1x1 value -- either true or false
    // x is (D+1)x1 array -- the pixels of the image as an array - D is number of features - +1 because of bias
    // w is (D+1)x1 array -- we automatically assume the transpose
    // y_hat is by definition wT*x = y_hat
    // Since our convention has w as Dx1 we can just use dot product
    // The bias is a 1x1 value

    // Calculate y_hat - This is 1x1 value
    double y_hat = logistic(dot_product(x, w, FEATURE_SIZE_W_BIAS));

    // Calculate difference between y_hat and y
    double y_diff = y_hat - y;

    // Calculate the gradient vector -- this is a (D+1)x1 sized vector
    for (int i = 0; i < FEATURE_SIZE_W_BIAS; i++){
        gradient_output[i] = x[i] * y_diff;
    }

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
    for (int i = 0; i < FEATURE_SIZE_W_BIAS; i++){
        w[i] = generate_random_weight();
    }
}


// Run Gradient Descent
void run_gradient_descent(double learning_rate, int number_to_classify, struct image_data* datum, double* w){

    // Step 1: Add a bias column to the feature vector - ONLY USE x_with_bias from here
    double x[FEATURE_SIZE_W_BIAS];
    add_bias_column(datum->data, x);

    double y = datum->label == number_to_classify ? 1 : 0;

    // Calculate the gradient
    double gradient_output[FEATURE_SIZE_W_BIAS];
    gradient(gradient_output, x, w, y);

    // Perform gradient descent -- aka update the weights based on the gradient
    for (int i = 0; i < FEATURE_SIZE_W_BIAS; i++){
        w[i]= w[i] - learning_rate * gradient_output[i];
    }

    // Step 3: Calculate the l2 norm in order to get the model error
    double error = calculate_L2_norm(gradient_output, FEATURE_SIZE_W_BIAS);

    // Step 4: Print out the error
    printf("The model error is: %lf\n", error);

}

int main() {
    struct image_data* data = read_data(MNIST_TRAINING_IMAGES, "training_sets/train-images.idx3-ubyte", "training_sets/train-labels.idx1-ubyte");

    double w[FEATURE_SIZE_W_BIAS];
    for (int i=0; i < FEATURE_SIZE_W_BIAS; i++)
        w[i] = generate_random_weight();

    for (int i=0; i < MNIST_TRAINING_IMAGES; i++)
        run_gradient_descent(LEARNING_RATE, NUMBER_TO_CLASSIFY, &data[i], w);
}
