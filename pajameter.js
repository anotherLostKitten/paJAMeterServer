japp nspc {
    required_linker_flags: -lm;
}


const parse_dataset = require('./parse_mnist_dataset.js');

jdata {
    struct gradient_update_t {
        int dataTag;
        unsigned long long int logicalId;
        int gradient[128];
    } gradients as uflow;
}

const boundedDelayMax = 1;

let logicalIdCount = 0;


let data = parse_dataset.loadTrainingData();

let inProgress = [];

const nodeDatas = new Map();
const dataInProgress = new Map();
let dataCount = 0;


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
            console.log(data.value);
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
            console.log(vec);
            vec.push(dataTag);

            dataInProgress.set(dataTag, vec);
            console.log("assigning data", dataTag, vec, "to", logicalId);
            return vec;
        } else
            await jsys.sleep(1000);
    }
    return [];
}

async function aggregateUpdates() {
    console.log("waiting to aggregate updates", data.length, dataInProgress.size);
    console.log("where did it go?");
    while (data.length > 0 || dataInProgress.size > 0) {
        console.log("waiting for gradient updates...");
        var gradient_updates = gradients.read();
        console.log(gradient_updates);
    }
}

if (jsys.type === "fog") {
    aggregateUpdates();
}
