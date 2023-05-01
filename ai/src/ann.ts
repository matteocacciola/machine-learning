import { layers, regularizers, sequential, Sequential, Optimizer, Tensor, tensor2d } from '@tensorflow/tfjs';
import { ActivationIdentifier } from '@tensorflow/tfjs-layers/dist/keras_format/activation_config';

interface MLPConfig {
  inputSize: number;
  outputSize: number;
  hiddenLayers: number[];
  hiddenActivation?: ActivationIdentifier;
  outputActivation?: ActivationIdentifier;
  regularization?: {
    l1?: number;
    l2?: number;
  };
  optimizer?: 'sgd' | 'momentum' | 'rmsprop' | 'adam' | 'adadelta' | 'adamax' | 'adagrad';
  loss?: string;
  metrics?: string[];
  batchSize?: number;
  epochs?: number;
  /* pruning?: boolean;
  quantization?: boolean; */
}

export class MlpAnn {
  private readonly inputSize: number;
  private readonly outputSize: number;
  private readonly hiddenLayers: number[];
  private readonly hiddenActivation: ActivationIdentifier;
  private readonly outputActivation: ActivationIdentifier;
  private readonly regularization: {
    l1?: number;
    l2?: number;
  };
  private readonly optimizer: string;
  private readonly loss: string;
  private readonly metrics: string[];
  private readonly batchSize: number;
  private readonly epochs: number;
  /* private readonly pruning: boolean;
  private readonly quantization: boolean; */

  private model: Sequential;

  constructor(config: MLPConfig) {
    this.inputSize = config.inputSize;
    this.outputSize = config.outputSize;
    this.hiddenLayers = config.hiddenLayers;
    this.hiddenActivation = config.hiddenActivation ?? 'relu';
    this.outputActivation = config.outputActivation ?? 'softmax';
    this.regularization = config.regularization ?? {};
    this.optimizer = config.optimizer ?? 'adam';
    this.loss = config.loss ?? 'categoricalCrossentropy';
    this.metrics = config.metrics ?? ['accuracy'];
    this.batchSize = config.batchSize ?? 32;
    this.epochs = config.epochs ?? 10;
    /* this.pruning = config.pruning ?? false;
    this.quantization = config.quantization ?? false; */

    this.model = sequential({ layers: this.buildLayers() });
    this.compileModel();
  }

  private buildLayers(): layers.Layer[] {
    const ls: layers.Layer[] = new Array<layers.Layer>();

    // Add input layer
    ls.push(layers.dense({
      units: this.hiddenLayers[0],
      inputShape: [this.inputSize],
      activation: this.hiddenActivation,
      kernelRegularizer: regularizers.l1l2({ l1: this.regularization.l1, l2: this.regularization.l2 }),
    }));

    // Add hidden layers
    for (let i = 1; i < this.hiddenLayers.length; i++) {
      ls.push(layers.dense({
        units: this.hiddenLayers[i],
        activation: this.hiddenActivation,
        kernelRegularizer: regularizers.l1l2({ l1: this.regularization.l1, l2: this.regularization.l2 }),
      }));
    }

    // Add output layer
    ls.push(layers.dense({
      units: this.outputSize,
      activation: this.outputActivation,
      kernelRegularizer: regularizers.l1l2({ l1: this.regularization.l1, l2: this.regularization.l2 }),
    }));

    return ls;
  }

  private compileModel() {
    // @ts-ignore
    const optimizer: string | Optimizer = tfTrain[this.optimizer]();
    this.model.compile({
      optimizer,
      loss: this.loss,
      metrics: this.metrics,
    });
  }

  private async trainBatch(inputs: Tensor, targets: Tensor) {
    const history = await this.model.fit(inputs, targets, {
      batchSize: this.batchSize,
      epochs: 1,
      shuffle: true,
      validationSplit: 0.1,
    });

    return history.history;
  }

  public async train(inputs: number[][], targets: number[][], debug?: boolean) {
    const xs = tensor2d(inputs);
    const ys = tensor2d(targets);

    for (let i = 0; i < this.epochs; i++) {
      const history = await this.trainBatch(xs, ys);
      debug && console.log(`Epoch ${i + 1}: ${JSON.stringify(history)}`);
    }

    xs.dispose();
    ys.dispose();

    /* if (this.pruning) {
      this.model = await prune(this.model);
    } */

    /* if (this.quantization) {
      this.model = await convertToTensorflow(this.model);
    } */
  }

  public evaluate(inputs: number[][]) {
    const xs = tensor2d(inputs);
    const outputs = this.model.predict(xs) as Tensor;
    const predictions = outputs.arraySync() as number[][];

    xs.dispose();
    outputs.dispose();

    return predictions;
  }
}