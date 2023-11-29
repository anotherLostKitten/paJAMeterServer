// Imports
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// Constants
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT
#define FEATURE_SIZE_W_BIAS MNIST_IMAGE_SIZE + 1

double calculate_L2_norm(double *vector, int size) {
    
    // init the double sum
    double sum_of_squares = 0.0;

    // sum the squares
    for (int i = 0; i < size; i++) {
        sum_of_squares += vector[i] * vector[i];
    }

    // calculate the square root
    return sqrt(sum_of_squares);
}

// Function to generate random weight between -0.5 and .5
double generate_random_weight() {
    return (((double) rand()) / ((double) RAND_MAX)) -.5 ;
}


// Logistic Function
double logistic(double z) {
    return 1 / (1 + exp(-z));
}

// Dot product function for vectors
double dot_product(double *a, double *b, int size) {
    double result = 0.0;

    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }

    return result;
}

void gradient(double *gradient_output, double *x, double *w, int *y){
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
    double y_diff = y_hat - *y;
    
    // Calculate the gradient vector -- this is a (D+1)x1 sized vector
    for (int i = 0; i < FEATURE_SIZE_W_BIAS; i++){
        gradient_output[i] = x[i]*(y_diff);
    }
    
}

// Initialize weights needed for gradient descent - add an extra row for bias
void init_weights(double *w){
    for (int i = 0; i < FEATURE_SIZE_W_BIAS; i++){
        w[i] = generate_random_weight();
    }
}


// Run Gradient Descent
void run_gradient_descent(double *learning_rate, double *x, int *y, double *w){
    
    // Step 1: Calculate the gradient
    double* gradient_output = malloc(FEATURE_SIZE_W_BIAS * sizeof(double));
    gradient(gradient_output, x, w, y);
    
    // Step 2: Perform gradient descent -- aka update the weights based on the gradient
    for (int i = 0; i < FEATURE_SIZE_W_BIAS; i++){
        w[i]= w[i] - (*learning_rate)* (gradient_output[i]);
    }
    
    // Step 3: Calculate the l2 norm in order to get the model error
    double error = calculate_L2_norm(gradient_output, FEATURE_SIZE_W_BIAS);
    
    // Step 4: Print out the error
    printf("The model error is: %lf\n", error);
    
    // Step 5: Free the malloced variable
    free(gradient_output);
 
}

// Code to initalize binary classifier -- runs 100 iterations of gradient descent one image at a time
void initalize_classifier(double *learning_rate, int number_to_classify, double *x, int *y, int image_index){
    
    // Run grad descent on 100 images
    for (int i = image_index; i < 100+image_index; i++){
        
        // Transform x feature to proper size of image and add the bias -> (D+1)x1
        int* x_transform = malloc(sizeof(double)*FEATURE_SIZE_W_BIAS);
        
        // Add bias and x values to x_transform
        for (int j = 0; j < FEATURE_SIZE_W_BIAS; j++) {
            if (j == 0){
                x_transform[j] = 1;
            }else{
               x_transform[j] = x[(i*MNIST_IMAGE_SIZE)+(j)];
            }
        }
        
        // Transform y label to binary -> 1x1 binary value
        int* y_transform = malloc(sizeof(int));
        *y_transform = (*y[i] == number_to_classify) ? 1 : 0;
        
        // Run grad descent
        run_gradient_descent(learning_rate, x_transform, y_transform, w);
        
        // Free the malloced variables
        free(x_transform);
        free(y_transform);   
    }
}

// int main(){
    
// }