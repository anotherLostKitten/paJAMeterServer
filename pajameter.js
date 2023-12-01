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
const heartbeats = new Map();
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
        var lidhandle = getLogicalId(jsys.id);
        try {
            var logicalId = await lidhandle.next();
            if (logicalId.value !== 0) {
                lidhandle.return();
                return logicalId.value;
            }
        } catch(e) {
            console.log(e.message, "... retrying");
        }
        await jsys.sleep(100);
    }
}

const devids = new Set();

jsync unsigned long long int {fogOnly} getLogicalId(cid: char*) {
    if (devids.has(cid))
        return 0;
    devids.add(cid);
    let lid = ++logicalIdCount;
    console.log("registered node with logical id", lid);

    nodeDatas.set(lid, new Set());
    heartbeats.set(lid, setTimeout(() => {
        console.log("device", lid, "timed out");
        for (var e of nodeDatas.get(lid)) {
            if (dataInProgress.has(e)) {
                var vec = dataInProgress.get(e);
                dataInProgress.delete(e);

                vec.pop(); // datatag
                var label = vec.pop();
                data.push([label, vec]);
            }
        }

        nodeDatas.get(lid).clear();
    }, 1000));

    return lid;
}

let datacount = 0;

jsync int[800] {deviceOnly} getNextDataLocal(logicalId: int) {
    while(1) {
        var dathandle = getNextData(logicalId);
        try {
            var data = await dathandle.next(logicalId);
            dathandle.return();
            // console.log("passing data value", data.value.at(-1));
            if (++datacount % 100 == 0)
                console.log("processed", datacount, "data items");
            return data.value;
        } catch(e) {
            console.log(e.message, "... retrying");
        }
        await jsys.sleep(100);
    }
}

jsync int[800] {fogOnly} getNextData(logicalId: int) {
    if (!nodeDatas.has(logicalId)) {
        console.log("logical id", logicalId, "not recognized, stopping device");
        return [];
    }
    if (data.length > 0) {
        if (nodeDatas.get(logicalId).size < boundedDelayMax) {
            let dataTag = ++dataCount;

            nodeDatas.get(logicalId).add(dataTag);
            heartbeats.get(logicalId).refresh();

            let vec = data.pop();
            vec[1].push(vec[0]);
            vec = vec[1];
            // console.log(vec);
            vec.push(dataTag);

            dataInProgress.set(dataTag, vec);
            // console.log("assigning data", dataTag, "to", logicalId);
            return vec;
        } else {
            await jsys.sleep(100);

            if (nodeDatas.get(logicalId).size > 0)
                for (var e of nodeDatas.get(logicalId))
                    return dataInProgress.get(e);
            return [0];
        }
    }
    if (nodeDatas.get(logicalId).size > 0)
        for (var e of nodeDatas.get(logicalId))
            return dataInProgress.get(e);

    return [];
}

function initModel() {
    for (var i = 0; i < 7850; i++)
        weights[i] = (Math.random() - 0.5);
    model.write(weights);
}



async function applyGradients(gradient_vec) {
    for (var g of gradient_vec) {
        for (var i = 0; i < g.length; i++) {
            weights[i] -= g[i] * LEARNING_RATE;
        }
    }
}

async function aggregateUpdates() {
    while (data.length > 0 || dataInProgress.size > 0) {
        // console.log("waiting to aggregate updates", data.length, dataInProgress.size);
        var gradient_updates = await gradients.readLast();
        // console.log(gradient_updates);
        gradient_vec = [];

        if (!Array.isArray(gradient_updates)) {
            gradient_updates = [gradient_updates];
        }
        for (var gradient of gradient_updates) {
            if (nodeDatas.has(gradient.logicalId) && dataInProgress.has(gradient.dataTag)) {
                nodeDatas.get(gradient.logicalId).delete(gradient.dataTag);
                heartbeats.get(gradient.logicalId).refresh();
                dataInProgress.delete(gradient.dataTag);
                gradient_vec.push(gradient.gradient);
            }
        }
        applyGradients(gradient_vec);
    }

    console.log("done; testing");
    let test_set = parse_dataset.loadTestingData();

    let success = 0;
    for (var img of test_set) {
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
        console.log("updating model");
        model.write(weights);
    }, 300);
    await jsys.sleep(100);
    aggregateUpdates();
}
