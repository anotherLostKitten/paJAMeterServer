const fs = require('fs');

function parseMNISTDataset(setSize, imagesPath, labelsPath, outputPath) {
    const dataFileBuffer = fs.readFileSync(imagesPath);
    const labelFileBuffer = fs.readFileSync(labelsPath);
    const pixelValues = [];

    for (let image = 0; image < setSize; image++) {
        const pixels = [];

        for (let x = 0; x < 28; x++) {
            for (let y = 0; y < 28; y++) {
                pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16]);
            }
        }

        const imageData = {};
        imageData[labelFileBuffer[image + 8]] = pixels; // Assuming labelFileBuffer contains label values directly

        pixelValues.push(imageData);
    }

    fs.writeFileSync(outputPath, JSON.stringify(pixelValues));
}


const trainingSetSize = 60000; // Number of examples in the training set
const testSetSize = 10000; // Number of examples in the test set
const trainingImagesPath = __dirname + '/training_sets/train-images.idx3-ubyte';
const trainingLabelsPath = __dirname + '/training_sets/train-labels.idx1-ubyte';
const testImagesPath = __dirname + '/training_sets/t10k-images.idx3-ubyte';
const testLabelsPath = __dirname + '/training_sets/t10k-labels.idx1-ubyte';

// Generate parsed JSON for training set
parseMNISTDataset(trainingSetSize, trainingImagesPath, trainingLabelsPath, __dirname + '/training_sets/parsed_mnist_training_set.json');

// Generate parsed JSON for test set
parseMNISTDataset(testSetSize, testImagesPath, testLabelsPath, __dirname + '/training_sets/parsed_mnist_test_set.json');

// The parsed data looks like:
//  '5': [
//   0, 0, 0, 0, 0, 0, 0, 0, ...