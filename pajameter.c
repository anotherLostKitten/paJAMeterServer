unsigned long long int logicalId = 0;
int hasData = 1;

jasync calculateGradient(int data[], int label, int dataTag) {
    printf("computing gradient for data %d (label: %d)\n", dataTag, label);

    jsys.sleep(100000); // TODO

    jarray int gradient[128] = {1, 2, 3};

    printf("writing to uflow %d\n", dataTag);
    struct gradient_update_t gradient_wrapper = {.dataTag = dataTag, .logicalId = logicalId, .gradient = gradient};
    gradients.write(&gradient_wrapper);
}


jasync dataFetch() {
    logicalId = getLogicalIdLocal();
    printf("got logical id %d\n", logicalId);

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
