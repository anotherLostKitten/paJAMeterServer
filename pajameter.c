#define BOUNDED_DELAY 4

jarray int owned_datas[BOUNDED_DELAY] = {0};
jarray int recentq[8] = {0};
int recentqcirc = 0;


jarray double weights[7850];

unsigned long long int logicalId = 0;
int hasData = 1;


jasync calculateGradient(int data[], int label, int dataTag) {
    struct gradient_update_t* gradient_wrapper = malloc(sizeof(struct gradient_update_t));
    gradient_wrapper->dataTag = dataTag;
    gradient_wrapper->logicalId = logicalId;
    nvoid_init((nvoid_t*)&gradient_wrapper->gradient, 7850, 'd', NULL, 0);

    double yhat[10], sum = 0;

    for (int n = 0; n < 10; n++) {
        double dp = weights.data[n * 785 + 784];
        for (int i = 0; i < 784; i++) {
            jsys.dontyield();
            dp += weights.data[n * 785 + i] * (double)data->data[i] / 256.0;
            // if (isnan(weights.data[n * 785 + i]))
            //     printf("nan from weights %d\n", n * 785 + i);
        }
        sum += yhat[n] = exp(dp);
        jsys.yield();
    }

    double l2_norm_sum = 0.0;

    for (int n = 0; n < 10; n++) {
        double ydif = yhat[n] / sum - (n == label ? 1.0 : 0.0);

        for (int i = 0; i < 784; i++) {
            jsys.dontyield();
            gradient_wrapper->gradient.data[n * 785 + i] = ydif * (double)data->data[i] / 256.0;
            l2_norm_sum += gradient_wrapper->gradient.data[n * 785 + i] * gradient_wrapper->gradient.data[n * 785 + i];
            // if (isnan(gradient_wrapper->gradient.data[n * 785 + i]))
            //     printf("nan from gradient %d\n", n * 785 + i);
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
    if (recentq.len < 2 * BOUNDED_DELAY) {
        recentq.data[recentq.len++] = dataTag;
    } else {
        recentq.data[recentqcirc++] = dataTag;
        recentqcirc %= 2 * BOUNDED_DELAY;
    }

    printf("computed gradient for data %d (label: %d), error %lf\n", dataTag, label, sqrt(l2_norm_sum));

    gradients.write(gradient_wrapper);
    free(gradient_wrapper);
}

jasync pollUpdates() {
    while (hasData) {
        jsys.sleep(300000);
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
                if (owned_datas.data[i] == dataTag || recentq.data[i] == dataTag || recentq.data[i + BOUNDED_DELAY] == dataTag)
                    goto skip;
            }
            for (int i = 0; i < BOUNDED_DELAY; i++) {
                jsys.dontyield();
                if (owned_datas.data[i] == 0) {
                    owned_datas.data[i] = dataTag;
                    break;
                }
            }

            calculateGradient(&data, label, dataTag);
        } else if (data.len == 0)
            hasData = 0;
    skip:jsys.sleep(100);
    }
}


int main(int argc, char* argv[]) {
    dataFetch();

    return 0;
}
