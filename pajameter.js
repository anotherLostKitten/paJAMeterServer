japp nspc {
    required_clibs: math.h, float.h;
    required_linker_flags: -lm;
}


const parse_dataset = require('./parse_mnist_dataset.js');

jdata {
    struct gradient_update_t {
        int dataTag;
        unsigned long long int logicalId;
        double gradient[7850];
    } gradients as uflow;

    double model[7850] as dflow;
}

let weights = new Array(7850);

const boundedDelayMax = 4;

let logicalIdCount = 0;


let data;

let inProgress = [];

const nodeDatas = new Map();
const dataInProgress = new Map();
let dataCount = 0;

const LEARNING_RATE = 0.005;

jcond {
    fogOnly(me, you) {
        return jsys.type === "fog";
    }
    deviceOnly(me, you) {
        return jsys.type === "device";
    }
}

jsync unsigned long long int {deviceOnly} getLogicalIdLocal() {
    while(1) {
        var lidhandle = getLogicalId();
        try {
            var logicalId = await lidhandle.next();
            lidhandle.return();
            return logicalId.value;
        } catch(e) {
            console.log(e.message, "... retrying");
        }
        await jsys.sleep(100);
    }
}


jsync unsigned long long int {fogOnly} getLogicalId() {
    logicalIdCount++;
    console.log("registered node with logical id", logicalIdCount);

    nodeDatas.set(logicalIdCount, new Set());

    return logicalIdCount;
}

jsync int[800] {deviceOnly} getNextDataLocal(logicalId: int) {
    while(1) {
        var dathandle = getNextData(logicalId);
        try {
            var data = await dathandle.next(logicalId);
            dathandle.return();
            console.log("passing data value", data.value.at(-1));
            return data.value;
        } catch(e) {
            console.log(e.message, "... retrying");
        }
        await jsys.sleep(100);
    }
}

jsync int[800] {fogOnly} getNextData(logicalId: int) {
    if (data.length > 0) {
        if (nodeDatas.get(logicalId).size < boundedDelayMax) {
            let dataTag = dataCount++;

            nodeDatas.get(logicalId).add(dataTag);

            let vec = data.pop();
            vec[1].push(vec[0]);
            vec = vec[1];
            // console.log(vec);
            vec.push(dataTag);

            dataInProgress.set(dataTag, vec);
            console.log("assigning data", dataTag, "to", logicalId);
            return vec;
        } else
            await jsys.sleep(1000);
    }
    return [];
}

function initModel() {
    for (var i = 0; i < 7580; i++)
        weights[i] = (Math.random() - 0.5);
    model.write(weights);
}



async function applyGradients(gradient_vec) {
    for (var g in gradient_vec)
        for (var i = 0; i < g.length; i++)
            weights[i] -= g[i] * LEARNING_RATE;
}

async function aggregateUpdates() {
    while (data.length > 0 || dataInProgress.size > 0) {
        console.log("waiting to aggregate updates", data.length, dataInProgress.size);
        var gradient_updates = await gradients.readLast();
        // console.log(gradient_updates);
        gradient_vec = [];
        if (!Array.isArray(gradient_updates)) {
            nodeDatas.get(gradient_updates.logicalId).delete(gradient_updates.dataTag);
            dataInProgress.delete(gradient_updates.dataTag);
            gradient_vec.push(gradient_updates.gradient);
        } else
            for (var gradient in gradient_updates) {
                nodeDatas.get(gradient.logicalId).delete(gradient.dataTag);
                dataInProgress.delete(gradient.dataTag);
                gradient_vec.push(gradient.gradient);
            }
        applyGradients(gradient_vec);
    }

    console.log("done; testing");
    let test_set = parse_dataset.loadTestingData();


    let success = 0;
    for (var img in test_set) {
        var maxc = 0, maxn = -1;
        for (var n = 0; n < 10; n++) {
            var conf = weights[785 * n + 784];
            for (var i = 0; i < 784; i++)
                conf += weights[785 * n + i] * img[1][i] / 256.0;

            if (conf > maxc) {
                maxc = conf;
                maxn = n;
            }
        }
        if (maxn == img[0])
            success++;
    }
    console.log("accuracy: " + success + " / " + test_set.length + " (" + (success / test_set.length * 100.0) + "%)");
}

if (jsys.type === "fog") {
    data = parse_dataset.loadTrainingData();
    initModel();
    setInterval(() => {
        model.write(weights);
    }, 1000);
    await jsys.sleep(100);
    aggregateUpdates();
}
