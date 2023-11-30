jarray double weights[7850];

unsigned long long int logicalId = 0;
int hasData = 1;

jasync calculateGradient(int data[], int label, int dataTag) {
    printf("computing gradient for data %d (label: %d)\n", dataTag, label);

    jsys.sleep(100000); // TODO

    double gradient[3] = {1, 2, 3};

    printf("writing to uflow %d\n", dataTag);
    struct gradient_update_t* gradient_wrapper = malloc(sizeof(struct gradient_update_t));
    gradient_wrapper->dataTag = dataTag;
    gradient_wrapper->logicalId = logicalId;
    nvoid_init(&gradient_wrapper->gradient, 7850, 'd', gradient, 3);
    gradients.write(gradient_wrapper);
}


jasync dataFetch() {
    logicalId = getLogicalIdLocal();
    printf("got logical id %llu\n", logicalId);

    model.read(&weights);
    printf("got initial model\n");

    jarray int data[800];
    while (hasData) {
        data = getNextDataLocal(logicalId);
        if (data.len) {
            int dataTag = data.data[--data.len];
            int label = data.data[--data.len];
            calculateGradient(&data, label, dataTag);
        }
    }
}


int main(int argc, char* argv[]) {
    dataFetch();

    return 0;
}
