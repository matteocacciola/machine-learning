import { cond, constant, stubTrue } from 'lodash';

interface KernelFunction {
  (x1: number[], x2: number[], args?: any): number;
}

enum Kernels {
  linear = 'linear',
  poly = 'poly',
  rbf = 'rbf'
}

type KernelType = keyof typeof Kernels;
type Kernel = KernelFunction | KernelType;

const linearKernel: KernelFunction = (
  x1: number[],
  x2: number[]
): number => x1.reduce((acc, item, index) => acc + (item * x2[index]), 0);

const polyKernel: KernelFunction = (
  x1: number[],
  x2: number[],
  { gamma, degree }: { gamma: number, degree: number }
): number => Math.pow(gamma * polyKernel(x1, x2, { gamma, degree }) + 1, degree);

const rbfKernel: KernelFunction = (
  x1: number[],
  x2: number[],
  { gamma }: { gamma: number }
) => Math.exp(-gamma * x1.reduce((acc, item, index) => acc + Math.pow(item - x2[index], 2), 0));

class SupportVectorMachine {
  private readonly C: number;
  private kernel: KernelFunction;
  private readonly tol: number;
  private readonly maxIter: number;
  // @ts-ignore
  private kernelCache: number[][];
  // @ts-ignore
  private supportVectors: number[][];
  // @ts-ignore
  private supportVectorLabels: number[];
  // @ts-ignore
  private supportVectorWeights: number[];
  // @ts-ignore
  private b: number;

  constructor(C: number, kernel: Kernel, tol: number = 1e-4, maxIter: number = 100) {
    this.C = C;
    this.kernel = this.assignKernel(kernel);
    this.tol = tol;
    this.maxIter = maxIter;
  }

  private assignKernel(kernel: Kernel): KernelFunction {
    const c = cond<Kernel, KernelFunction>([
      [(k) => k === Kernels.linear, constant<KernelFunction>(linearKernel)],
      [(k) => k === Kernels.poly, constant<KernelFunction>(polyKernel)],
      [(k) => k === Kernels.rbf, constant<KernelFunction>(rbfKernel)],
      [stubTrue, constant<KernelFunction>(kernel as KernelFunction)]
    ]);

    return c(kernel);
  }

  private computeKernelMatrix(X: number[][]): number[][] {
    const n = X.length;
    const K = new Array(n);
    for (let i = 0; i < n; i++) {
      K[i] = new Array(n);
      for (let j = 0; j < n; j++) {
        K[i][j] = this.computeKernel(X[i], X[j]);
      }
    }
    return K;
  }

  private train(X: number[][], y: number[]) {
    const n = X.length;
    const K = this.computeKernelMatrix(X);

    const alpha = new Array(n).map(() => 0);
    this.supportVectors = [];
    this.supportVectorLabels = [];
    this.supportVectorWeights = new Array(n).map(() => 0);
    this.kernelCache = new Array(n);
    for (let i = 0; i < n; i++) {
      this.kernelCache[i] = new Array(n);
    }
    this.b = 0;

    let passes = 0;
    while (passes < this.maxIter) {
      let numChangedAlphas = 0;
      for (let i = 0; i < n; i++) {
        const Ei = this.classify(X[i]) - y[i];
        if ((y[i] * Ei < -this.tol && alpha[i] < this.C) || (y[i] * Ei > this.tol && alpha[i] > 0)) {
          let j;
          do {
            j = Math.floor(Math.random() * n);
          } while (j === i);

          const Ej = this.classify(X[j]) - y[j];

          const ai = alpha[i];
          const aj = alpha[j];
          const C = this.C;
          const y1 = y[i];
          const y2 = y[j];
          const k11 = K[i][i];
          const k22 = K[j][j];
          const k12 = K[i][j];
          const eta = k11 + k22 - 2 * k12;

          let L, H;
          if (y1 === y2) {
            L = Math.max(0, ai + aj - C);
            H = Math.min(C, ai + aj);
          } else {
            L = Math.max(0, aj - ai);
            H = Math.min(C, C + aj - ai);
          }

          if (L === H) {
            continue;
          }

          const eta2 = 2 * eta;
          const y1y2 = y1 * y2;
          const s = y1y2 * eta - ai * y1y2 * eta - y1 * (aj - ai) * k11 - y2 * (aj - ai) * k12;
          alpha[j] -= y2 * s / eta2;

          if (alpha[j] > H) {
            alpha[j] = H;
          } else if (alpha[j] < L) {
            alpha[j] = L;
          }

          if (Math.abs(alpha[j] - aj) < this.tol) {
            continue;
          }

          alpha[i] += y1y2 * (aj - alpha[j]);

          const b1 = this.b - Ei - y1 * (alpha[i] - ai) * k11 - y2 * (alpha[j] - aj) * k12;
          const b2 = this.b - Ej - y1 * (alpha[i] - ai) * k12 - y2 * (alpha[j] - aj) * k22;

          if (alpha[i] > 0 && alpha[i] < this.C) {
            this.b = b1;
          } else if (alpha[j] > 0 && alpha[j] < this.C) {
            this.b = b2;
          } else {
            this.b = (b1 + b2) / 2;
          }

          numChangedAlphas++;
        }
      }

      passes = numChangedAlphas === 0  ? passes + 1 : 0;
    }

    alpha.map((a, i) => {
      if (a > 0) {
        this.supportVectors.push(X[i]);
        this.supportVectorLabels.push(y[i]);
        this.supportVectorWeights[i] = y[i] * a;
      }
    });
  }

  private computeKernel(x1: number[], x2: number[]): number {
    const i = x1[0];
    const j = x2[0];
    if (this.kernelCache[i][j] !== null) {
      return this.kernelCache[i][j];
    }
    const k = this.kernel(x1.slice(1), x2.slice(1));
    this.kernelCache[i][j] = k;
    this.kernelCache[j][i] = k;
    return k;
  }

  private classify(x: number[]): number {
    const prediction = this.supportVectors.reduce((acc, item, i) => {
      const kernelValue = this.computeKernel([i, ...item], [this.supportVectorWeights[i], ...x]);
      acc += this.supportVectorWeights[i] * this.supportVectorLabels[i] * kernelValue;
      return acc;
    }, 0);
    return prediction - this.b;
  }

  public fit(X: number[][], y: number[]) {
    this.train(X, y);
  }

  public evaluate(X: number[][]): number[] {
    return X.map((item) => this.classify(item));
  }
}