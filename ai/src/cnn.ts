import { layers, LayersModel, ModelFitArgs, History, sequential, Tensor, tidy, Tensor1D } from '@tensorflow/tfjs';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

enum LayerConfigEnum {
  conv = 'conv',
  pool = 'pool',
  dense = 'dense',
}

type LayerConfigType = keyof typeof LayerConfigEnum;

interface LayerConfig {
  type: LayerConfigType;
  filters?: number;
  kernelSize?: number | number[];
  strides?: number[];
  activation?: ActivationIdentifier;
  poolSize?: number | [number, number];
  poolStrides?: number | [number, number];
  units?: number;
}

export class ConvolutionaNeuralNetwork {
  private model: LayersModel;

  constructor(
    private readonly inputShape: number[],
    private readonly numClasses: number,
    private readonly layerConfigs: LayerConfig[]
  ) {
    const layers = this.buildModel();

    this.model = sequential({ layers });
    this.model.summary();
    this.model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  }

  private addConvLayer(layerConfig: LayerConfig, index: number): layers.Layer {
    const { filters, kernelSize, activation, strides } = layerConfig;
    if (!filters || !kernelSize) {
      throw new Error('Missing parameters for convolutional layer');
    }
    return layers.conv2d({
      filters,
      kernelSize,
      strides,
      activation,
      ...(index === 0 && { inputShape: this.inputShape })
    });
  }

  private addPoolLayer(layerConfig: LayerConfig): layers.Layer {
    const { poolSize, poolStrides: strides } = layerConfig;
    return layers.maxPooling2d({ poolSize, strides });
  }

  private addDenseLayer(layerConfig: LayerConfig): layers.Layer {
    const { units, activation } = layerConfig;
    if (!units) {
      throw new Error('Missing parameters for dense layer');
    }
    return layers.dense({ units, activation });
  }

  private buildModel(): layers.Layer[] {
    const ls: layers.Layer[] = this.layerConfigs.map((layerConfig, index) => {
      switch (layerConfig.type) {
        case LayerConfigEnum.conv:
          return this.addConvLayer(layerConfig, index);
        case LayerConfigEnum.pool:
          return this.addPoolLayer(layerConfig);
        case LayerConfigEnum.dense:
          return this.addDenseLayer(layerConfig);
        default:
          throw new Error('Invalid layer type');
      }
    })

    ls.push(layers.flatten());
    ls.push(layers.dense({ units: this.numClasses, activation: 'softmax' }));

    return ls;
  }

  public async train(xTrain: Tensor, yTrain: Tensor, args: ModelFitArgs): Promise<History> {
    return await this.model.fit(xTrain, yTrain, args);
  }

  public evaluate(x: Tensor): Tensor1D {
    return tidy(() => {
      const output = this.model.predict(x) as Tensor;
      return output.argMax(-1);
    });
  }
}

/**
 * EXAMPLE OF USAGE
 *
 * Here we use the MNIST dataset. It is a popular benchmark dataset in the field of machine learning and computer vision.
 * It consists of a set of 70,000 small images of handwritten digits, divided into 60,000 training examples and 10,000
 * test examples. Each image is grayscale and has a resolution of 28 x 28 pixels. The goal of the MNIST dataset is to
 * classify the digit in each image correctly, with a high degree of accuracy. The dataset has been widely used to develop
 * and test various machine learning algorithms, including deep neural networks, and has played a key role in advancing the
 * field of computer vision.

import { data, oneHot } from '@tensorflow/tfjs';

const TRAIN_DATA_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/train.csv';
const TEST_DATA_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/test.csv';
const NUM_TRAIN_EXAMPLES = 60000;
const NUM_CLASSES = 10; // (since there are 10 possible digits, 0-9)
const NUM_TEST_EXAMPLES = 10000;

const loadMnistData = async (): Promise<[Tensor, Tensor, Tensor, Tensor]> => {
  const trainData = await data.csv(TRAIN_DATA_URL, { hasHeader: false });
  const testData = await data.csv(TEST_DATA_URL, { hasHeader: false });

  const [xTrain, yTrain, xTest, yTest] = await Promise.all([
    trainData
      .map((row: any) => row.slice(1))
      .batch(NUM_TRAIN_EXAMPLES)
      .map((batch: any) => batch.toFloat().div(255))
      .toArray(),
    trainData
      .map((row: any) => row.slice(0, 1))
      .batch(NUM_TRAIN_EXAMPLES)
      .map((batch: any) => oneHot(batch, NUM_CLASSES))
      .toArray(),
    testData
      .map((row: any) => row.slice(1))
      .batch(NUM_TEST_EXAMPLES)
      .map((batch: any) => batch.toFloat().div(255))
      .toArray(),
    testData
      .map((row: any) => row.slice(0, 1))
      .batch(NUM_TEST_EXAMPLES)
      .map((batch: any) => oneHot(batch, NUM_CLASSES))
      .toArray()
  ]);

  return [xTrain[0], yTrain[0], xTest[0], yTest[0]];
}

(async () => {
  const NUM_EPOCHS = 10;
  const BATCH_SIZE = 128;

  const inputShape = [28, 28, 1];
  const numClasses = 10;
  const layerConfigs: LayerConfig[] = [
    { type: 'conv', filters: 32, kernelSize: 3, activation: 'relu' },
    { type: 'pool', poolSize: 2 },
    { type: 'conv', filters: 64, kernelSize: 3, activation: 'relu' },
    { type: 'pool', poolSize: 2 },
    { type: 'dense', units: 128, activation: 'relu' },
    { type: 'dense', units: 10, activation: 'softmax' }
  ];

  const cnn = new ConvolutionaNeuralNetwork(inputShape, numClasses, layerConfigs);

  const [xTrain, yTrain, xTest, yTest] = await loadMnistData();

  await cnn.train(xTrain, yTrain, {
    batchSize: BATCH_SIZE,
    epochs: NUM_EPOCHS,
    validationData: [xTest, yTest]
  });

  const testAccuracy = cnn.evaluate(xTest);
  console.log(`Test accuracy: ${testAccuracy}`);
})();

 */