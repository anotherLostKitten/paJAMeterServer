#define BOUNDED_DELAY 4

jarray int owned_datas[BOUNDED_DELAY];

jarray double weights[7850];

unsigned long long int logicalId = 0;
int hasData = 1;

jasync calculateGradient(int data[], int label, int dataTag) {
    printf("computing gradient for data %d (label: %d)\n", dataTag, label);

    // printf("writing to uflow %d\n", dataTag);
    struct gradient_update_t* gradient_wrapper = malloc(sizeof(struct gradient_update_t));
    gradient_wrapper->dataTag = dataTag;
    gradient_wrapper->logicalId = logicalId;
    nvoid_init((nvoid_t*)&gradient_wrapper->gradient, 7850, 'd', NULL, 0);

    double yhat[10], sum = 0;

    for (int n = 0; n < 10; n++) {
        double dp = weights.data[n * 785 + 784];
        for (int i = 0; i < 784; i++) {
            jsys.dontyield();
            dp +=  (double)data->data[i] / 256.0 * weights.data[n * 785 + i];
        }
        sum += yhat[n] = exp(dp);
        jsys.yield();
    }
    for (int n = 0; n < 10; n++) {
        double ydif = yhat[n] / sum - (n == label ? 1.0 : 0.0);

        for (int i = 0; i < 784; i++) {
            jsys.dontyield();
            gradient_wrapper->gradient.data[n * 785 + i] = (double)data->data[i] / 256.0 * ydif;
        }
        gradient_wrapper->gradient.data[n * 785 + 784] = ydif;
        jsys.yield();
    }
    gradient_wrapper->gradient.len = 7850;

    for (int i = 0; i < BOUNDED_DELAY; i++) {
        jsys.dontyield();
        if (owned_datas.data[i] == dataTag)
            owned_datas.data[i] = 0;
    }

    gradients.write(gradient_wrapper);
}

jasync pollUpdates() {
    while (hasData) {
        jsys.sleep(50000);
        model.read(&weights);
        printf("updated model\n");
    }
}

jasync dataFetch() {
    logicalId = getLogicalIdLocal();
    printf("got logical id %llu\n", logicalId);

    model.read(&weights);
    pollUpdates();
    printf("got initial model\n");

    jarray int data[800];
    while (hasData) {
        data = getNextDataLocal(logicalId);
        if (data.len > 1) {
            int dataTag = data.data[--data.len];
            int label = data.data[--data.len];

            for (int i = 0, e = 0; i < BOUNDED_DELAY; i++) {
                jsys.dontyield();
                if (owned_datas.data[i] == dataTag)
                    goto skip;
                else if (owned_datas.data[i] == 0 && !e++)
                    owned_datas.data[i] = dataTag;
            }

            calculateGradient(&data, label, dataTag);
        } else if (data.len == 1)
            hasData = 0;
    skip:jsys.sleep(100);
    }
}


int main(int argc, char* argv[]) {
    dataFetch();

    return 0;
}
