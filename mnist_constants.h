#ifndef MNIST_CONSTANTS_H
#define MNIST_CONSTANTS_H
#include <stdint.h>

#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE (MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT)
#define FEATURE_SIZE_W_BIAS (MNIST_IMAGE_SIZE + 1)

#define MNIST_TRAINING_IMAGES 60000
#define MNIST_TESTING_IMAGES 10000
#define LEARNING_RATE 0.005
#define NUMBER_TO_CLASSIFY 1

struct image_data {
    uint8_t data[MNIST_IMAGE_SIZE];
    uint8_t label;
} __attribute__ ((__packed__));

struct image_data* read_data(size_t set_size, char* img_path, char* label_path);

#endif
