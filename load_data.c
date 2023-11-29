#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "mnist_constants.h"

struct image_data* read_data(size_t set_size, char* img_path, char* lbl_path) {
    struct image_data* datas = malloc(sizeof(struct image_data) * set_size);

    FILE* img_f = fopen(img_path, "rb"),
        * lbl_f = fopen(lbl_path, "rb");

    assert(img_f);
    assert(lbl_f);

    fseek(img_f, 16, SEEK_SET);
    fseek(lbl_f, 8, SEEK_SET);

    for (int i = 0; i < set_size; i++) {
        fread(datas[i].data, MNIST_IMAGE_SIZE, 1, img_f);
        fread(&datas[i].label, 1, 1, lbl_f);
    }

    fclose(img_f);
    fclose(lbl_f);

    return datas;
}
