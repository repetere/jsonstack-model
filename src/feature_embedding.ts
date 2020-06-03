import { TensorScriptOptions, TensorScriptProperties, Matrix, Vector, TensorScriptLayers, NestedArray, InputTextArray, PredictionOptions, Shape, TensorScriptLSTMModelContext, LambdaLayer, DenseLayer, asyncForEach, Features, Corpus} from './model_interface';
import * as Tensorflow from '@tensorflow/tfjs-node';
import { BaseNeuralNetwork, } from './base_neural_network';
import range from 'lodash.range';
import TSNE from 'tsne-js';
//https://towardsdatascience.com/understanding-feature-engineering-part-4-deep-learning-methods-for-text-data-96c44370bbfa

/**
 * use a corpus to generate features from an embedding layer with Tensorflow
 * @class FeatureEmbedding
 * @implements {BaseNeuralNetwork}
 */
export class FeatureEmbedding extends BaseNeuralNetwork {
  layers?: TensorScriptLayers;
  featureToId?: any;
  IdToFeature?: any;
  featureIds?: any;
  numberOfFeatures?: any;
  loss?: number;
  // settings: TensorScriptOptions;
  static async getFeatureDataSet(this: any, { inputMatrixFeatures, PAD = 'PAD', }: { inputMatrixFeatures: Corpus; PAD?: string;}) {
    let featIndex = 1;
    const IdToFeature = { 0: this && this.settings && this.settings.PAD ? this.settings.PAD : PAD };
    const featureToId = inputMatrixFeatures.reduce((result, inputFeatureArray) => { 
      inputFeatureArray.forEach((inputFeature) => {
        if (!result[inputFeature]) {
          result[inputFeature] = featIndex;
          //@ts-ignore
          IdToFeature[featIndex] = inputFeature;
          featIndex++;
        }
      });
      return result;
    }, {
      [PAD]:0
    });
    //@ts-ignore
    const featureIds = inputMatrixFeatures.map(inputFeatureArray =>
      inputFeatureArray.map(inputFeature => featureToId[inputFeature])
    );
    // console.log('featureIds', featureIds);
    // console.log('featureIds.length', featureIds.length);
    // console.log('inputMatrixFeatures.length', inputMatrixFeatures.length);
    const numberOfFeatures = Object.keys(featureToId).length;
    return {
      featureToId, //word2id
      IdToFeature, //id2word
      featureIds, //wids
      numberOfFeatures, //vocab_size
    };
  }
  static getMergedArray(base:Vector = [], merger:Vector= [], append=false, truncate=true) {
    let arr = new Array().concat(base);
    if (append) arr.splice(base.length-merger.length,merger.length,...merger);
    else arr.splice(0, merger.length, ...merger);
    if (truncate && append) return arr.slice(-1 * base.length);
    else if (truncate) return arr.slice(0, base.length);
    else return arr;
  }
  /**
   */
  static async getContextPairs(this:any, { inputMatrix, numberOfFeatures, window_size = 2, tf, }: { inputMatrix: Matrix; numberOfFeatures: number; window_size?: number; tf?: any;}) {
    const tensorflow = this && this.tf ? this.tf : Tensorflow;
    const context_length = (this && this.settings && this.settings.windowSize ? this.settings.windowSize : window_size) * 2;
    const [emptyXVector, emptyYVector] = await Promise.all([
      tensorflow.zeros([context_length]).array(),
      tensorflow.zeros([numberOfFeatures]).array(),
    ]);
    const x:Matrix = [];
    const y:Matrix = [];
    inputMatrix.forEach((inputVector:Vector,inputVectorIndex:number) => {
      inputVector.forEach((word:number, index:number) => {
        if (word != 0) {
          const output = new Array().concat(emptyYVector);
          const inputMerger = new Array().concat(inputMatrix[inputVectorIndex]);
          inputMerger.splice(index, 1);
          const input = FeatureEmbedding.getMergedArray(emptyXVector, inputMerger,true);
          output[word] = 1;
          // x.push([[word],input]);
          x.push(input);
          y.push(output);
        }
      });
    });

    return {
      context_length,
      emptyXVector,
      emptyYVector,
      x,
      y,
    };
  }
  getMergedArray: typeof FeatureEmbedding.getMergedArray;
  getFeatureDataSet: typeof FeatureEmbedding.getFeatureDataSet;
  getContextPairs: typeof FeatureEmbedding.getContextPairs;
  constructor(options:TensorScriptOptions = {}, properties?:TensorScriptProperties) {
    const config = {
      layers: [],
      type: 'cbow',
      compile: {
        loss: 'categoricalCrossentropy',
        optimizer: 'rmsprop',
      },
      fit: {
        epochs: 15,
        batchSize: 1,
      },
      embedSize: 50,
      windowSize: 2,
      PAD: 'PAD',
      streamInputMatrix:true,
      ...options
    };
    super(config, properties);
    this.type = 'FeatureEmbedding';
    this.featureToId;
    this.IdToFeature;
    this.featureIds;
    this.numberOfFeatures;
    this.getMergedArray = FeatureEmbedding.getMergedArray;
    this.getFeatureDataSet = FeatureEmbedding.getFeatureDataSet.bind(this);
    this.getContextPairs = FeatureEmbedding.getContextPairs.bind(this);
    return this;
  }
  /**
   * Adds dense layers to tensorflow classification model
   * @override 
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @param {Array<Object>} layers - model dense layer parameters
   */
  generateLayers(this: FeatureEmbedding,x_matrix:Matrix,y_matrix:Matrix, layers?:TensorScriptLayers) {
    // const xShape = this.getInputShape(x_matrix);
    const yShape = [this.numberOfFeatures, this.settings.embedSize,];// this.getInputShape(y_matrix);
    this.yShape = yShape;
    // this.xShape = xShape;
    const denseLayers:TensorScriptLayers = [];
    if (layers) {
      denseLayers.push(...layers);
    } else {
      /**
       * 
       cbow = Sequential()
cbow.add(Embedding(input_dim=numberOfFeatures, output_dim=embed_size, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(numberOfFeatures, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
       */
      denseLayers.push({ units: this.numberOfFeatures, inputDim: this.numberOfFeatures, outputDim: this.settings.embedSize, inputLength: (this.settings.windowSize||2) * 2, });
      denseLayers.push({
        lambdaFunction: 'result = tf.mean(input,1,true)',
        lambdaOutputShape: [this.numberOfFeatures, this.settings.embedSize]
      });
      denseLayers.push({ units: this.numberOfFeatures, activation: 'softmax', });
    }
    this.layers = denseLayers;
    // console.log({ denseLayers });
    // this.tf.serialization.registerClass(lambdaLayer);
    this.model.add(this.tf.layers.embedding(denseLayers[0]));
    // this.model.add(new lambdaLayer(denseLayers[1]));
    this.model.add(this.tf.layers.flatten());
    this.model.add(this.tf.layers.dense(denseLayers[2]));
  }
  async trainOnBatch({ x_input_matrix, y_output_matrix, epoch, trainingLoss, }: { x_input_matrix: Matrix, y_output_matrix: Matrix, epoch: number, trainingLoss: number, }) {
    let loss = Infinity;
    if (this.settings.fit?.callbacks?.onEpochBegin) this.settings.fit?.callbacks?.onEpochBegin(epoch, { loss:trainingLoss });
    await asyncForEach(x_input_matrix, async (x_input:Vector, xIndex:number) => {
      if (this.settings.fit?.callbacks?.onBatchBegin) this.settings.fit?.callbacks?.onBatchBegin(xIndex, { loss:trainingLoss, });
      const y_output = y_output_matrix[xIndex];
      const xShape = this.getInputShape([x_input]);
      const xs = this.tf.tensor(x_input, xShape);
      const yShape = this.getInputShape([y_output]);
      const ys = this.tf.tensor(y_output, yShape);
      // const xdata = await xs.data()
      // console.log({ xs, xdata, xShape });
      loss = await this.model.trainOnBatch(xs, ys);
      if (this.settings.fit?.callbacks?.onYield) this.settings.fit?.callbacks?.onYield(epoch, xIndex, { loss });
      if (this.settings.fit?.callbacks?.onBatchEnd) this.settings.fit?.callbacks?.onBatchEnd(xIndex, { loss });
      // console.log({ x_input, xIndex, xShape, y_output, yShape, loss })
      xs.dispose();
      ys.dispose();
    });
    if (this.settings.fit?.callbacks?.onEpochEnd) this.settings.fit?.callbacks?.onEpochEnd(epoch, { loss });
    return {
      loss
    };
  }
  async generateBatch() {
    const preTransformedMatrix: Matrix = this.featureIds;
    const context_length = (this && this.settings && this.settings.windowSize ? this.settings.windowSize : 2) * 2;
    const [emptyXVector, emptyYVector] = await Promise.all([
      this.tf.zeros([context_length]).array(),
      this.tf.zeros([this.numberOfFeatures]).array(),
    ]);
    let x_input_matrix: Matrix = [];
    let y_output_matrix: Matrix = [];
    let epoch = 0;
    let trainingLoss = Infinity;
    await asyncForEach(preTransformedMatrix, async (inputVector: Vector, inputVectorIndex: number) => {
      await asyncForEach(inputVector, async (word:number, index:number) => {
        if (word != 0) {
          const output = new Array().concat(emptyYVector);
          const inputMerger = new Array().concat(preTransformedMatrix[inputVectorIndex]);
          inputMerger.splice(index, 1);
          const input = FeatureEmbedding.getMergedArray(emptyXVector, inputMerger,true);
          output[word] = 1;
          // x.push([[word],input]);
          // x.push(input);
          // y.push(output);
          x_input_matrix = [input];
          y_output_matrix = [output];
          // console.log({ input, output });
          const modelStatus = await this.trainOnBatch({ x_input_matrix, y_output_matrix, epoch, trainingLoss,  });
          trainingLoss = modelStatus.loss;
        }
      });
    });
    return {
      loss: trainingLoss,
    };
  }
  async train(x_matrix: Matrix|Corpus, y_matrix:Matrix, layers?: DenseLayer[]) {
    const featureEmbedDataSet = await this.getFeatureDataSet({ inputMatrixFeatures: x_matrix, });
    this.featureToId = featureEmbedDataSet.featureToId;
    this.IdToFeature = featureEmbedDataSet.IdToFeature;
    this.featureIds = featureEmbedDataSet.featureIds;
    this.numberOfFeatures = featureEmbedDataSet.numberOfFeatures;
    if (this.compiled === false) {
      this.model = this.tf.sequential();
      //@ts-ignore
      this.generateLayers.call(this, [], [], layers || this.layers,
        // x_test, y_test
      );
      this.model.compile(this.settings.compile);
      this.compiled = true;
    }
    let loss = Infinity;
    if (this.settings.fit?.callbacks?.onTrainBegin) this.settings.fit?.callbacks?.onTrainBegin({ loss });
    await asyncForEach(range(1, this.settings.fit?.epochs), async (epoch: number) => {
      if (this.settings.streamInputMatrix) {
        let modelStatus = await this.generateBatch();
        loss = modelStatus.loss;
      } else {
        const cxt = await this.getContextPairs({ tf:this.tf, numberOfFeatures: this.numberOfFeatures, inputMatrix: this.featureIds });
        const x_input_matrix = cxt.x;
        const y_output_matrix = cxt.y;
        let modelStatus = await this.trainOnBatch({ x_input_matrix, y_output_matrix, epoch, trainingLoss:loss, });
        loss = modelStatus.loss;
      }
    });
    if (this.settings.fit?.callbacks?.onTrainEnd) this.settings.fit?.callbacks?.onTrainEnd({ loss });
    this.loss = loss;

    this.trained = true;
    return this.model;
  }
  // async calculate(x_matrix: Matrix | Vector | InputTextArray) {
  async calculate() {
    return this.model.getWeights()[0];
  }
  // async predict(input_matrix: any[], options: PredictionOptions | undefined) {
  async predict(options: PredictionOptions = {}) {
    const predictions = await this.calculate();
    if (options.json === false) {
      return await predictions.data();
    } else {
      // console.log({predictions})
      const arr = await predictions.array()
      if (!this.yShape) throw new Error('Model is missing yShape');
      return this.reshape(arr, predictions.shape);
    }
  }
  /**
   * Converts matrix of layer weights into labeled features
   * @example
const weights = [
  [1.5,1,4,1.6,3.5],
  [4.3,3.2,5.5,6.5]
]
FeatureEmbeddingInstance.labelWeights(weights) //=> 
weights = {
  car:[1.5,1,4,1.6,3.5],
  boat:[4.3,3.2,5.5,6.5]
}
   */
  labelWeights(weights: Matrix) {
    return weights.reduce((result: { [index: string]: Vector;}, weight:Vector, index:number) => { 
      result[this.IdToFeature[index]] = weight;
      return result;
    }, {});
  }
  /**
   * Uses tSNE to reduce dimensionality of features
   * @example
const weights = [
  [1.5,1,4,1.6,3.5],
  [4.3,3.2,5.5,6.5]
]
FeatureEmbeddingInstance.reduceWeights(weights) //=> 
[
  [1,2],
  [2,3],
]
   */
  async reduceWeights(weights: Matrix, options?: any) {
    let model = new TSNE({
      dim: 2,
      perplexity: 30.0,
      earlyExaggeration: 4.0,
      learningRate: 100.0,
      nIter: 1000,
      metric: 'euclidean',
      ...options
    });
    model.init({
      data: weights,
      type: options ? options.type : 'dense'
    });
    let [error, iter] = model.run();
    // console.log({ error, iter });
    let output = model.getOutput(); 
    return output;
  }
/**
 * Uses either cosineProximity or Eucledian distance to rank similarity
@example
//weights = [ [1,2,3,], [1,2,2], [0,-1,3] ]
//labeledWeights = [ {car:[1,2,3,],tesla:[1,2,2],boat:[0,-1,3]}]
FeatureEmbeddingInstance.findSimilarFeatures(weights,{features:['car'], limit:2,}) //=> 
{
  car:[
  {
    comparedFeature: 'tesla',
    proximity: -0.5087087154388428,
    distance: 0.03015853278338909
  },
  {
    comparedFeature: 'boat',
    proximity: -0.3032159209251404,
    distance: 0.036241017282009125
  },
  ]
}
 */
  async findSimilarFeatures(weights: Matrix, options: SimilarFeatureOptions = {}) {
    const tf:typeof Tensorflow = this.tf;
    const { features = [], limit=5, labeledWeights, metric='distance' } = options;
    const labeledFeatureWeights = labeledWeights || this.labelWeights(weights);
    if(this.settings && this.settings.PAD) delete labeledFeatureWeights[this.settings.PAD]
    return features.reduce((result:SimilarFeatures,feature: string) => {
      const featureWeights = labeledFeatureWeights[feature];
      if(!featureWeights) throw new ReferenceError(`Invalid feature: ${feature}`);
     
      const sims = Object.keys(labeledFeatureWeights)
        .map(searchFeature => {
          const prox = tf.tidy(() => {
            const proximity = tf.metrics.cosineProximity(
              tf.tensor(featureWeights),
              tf.tensor(labeledFeatureWeights[searchFeature])
            );
            const distance = tf.metrics.meanSquaredError(
              tf.tensor(featureWeights),
              tf.tensor(labeledFeatureWeights[searchFeature])
            );
            return [
              proximity.asScalar().dataSync()[0],
              distance.asScalar().dataSync()[0],
            ];
          });
          return {
            comparedFeature: searchFeature,
            proximity: prox[0],
            distance: prox[1],
          }
        })
        .sort((a, b) => (metric === 'distance')
          ? a.distance - b.distance
          : a.proximity - b.proximity);
      // sims.shift()
      
      result[feature]= sims.slice(1, limit+1);
      return result;
    }, {});
  }
}

export type LabeledWeights = {
  [index: string]: Vector;
}
export enum SimilarityMetric {
  DISTANCE = 'distance',
  PROXIMITY = 'proximity',
}
export type SimilarFeatureOptions = {
  features?: string[];
  limit?: number;
  labeledWeights?: LabeledWeights; 
  metric?: SimilarityMetric;
}
export type SimilarFeatures =  { [index: string]: SimilarFeature[];} 
export type SimilarFeature = {
  comparedFeature: string;
  distance: number;
  proximity: number;
}