import { TensorScriptOptions, TensorScriptProperties, Matrix, Vector, TensorScriptLayers, NestedArray, InputTextArray, PredictionOptions, Shape, TensorScriptLSTMModelContext, lambdaLayer, DenseLayer, } from './model_interface';
import { BaseNeuralNetwork, } from './base_neural_network';
import range from 'lodash.range';

async function asyncForEach(array, callback) {
  for (let index = 0; index < array.length; index++) {
    await callback(array[index], index, array);
  }
}
/**
 * Long Short Term Memory Time Series with Tensorflow
 * @class FeatureEmbedding
 * @implements {BaseNeuralNetwork}
 */
export class FeatureEmbedding extends BaseNeuralNetwork {
  layers?: TensorScriptLayers;
  word2id?: any;
  id2word?: any;
  wids?: any;
  vocab_size?: any;
  // settings: TensorScriptOptions;
  static async getFeatureDataSet({ inputMatrixFeatures, PAD = 'PAD', }: { inputMatrixFeatures: Matrix; PAD?: string;}) {
    let featIndex = 1;
    const id2word = { 0: PAD };
    const word2id = inputMatrixFeatures.reduce((result, inputFeatureArray) => { 
      inputFeatureArray.forEach((inputFeature) => {
        if (!result[inputFeature]) {
          result[inputFeature] = featIndex;
          //@ts-ignore
          id2word[featIndex] = inputFeature;
          featIndex++;
        }
      });
      return result;
    }, {
      [PAD]:0
    });
    //@ts-ignore
    const wids = inputMatrixFeatures.map(inputFeatureArray =>
      inputFeatureArray.map(inputFeature => word2id[inputFeature])
    );
    // console.log('wids', wids);
    // console.log('wids.length', wids.length);
    // console.log('inputMatrixFeatures.length', inputMatrixFeatures.length);
    const vocab_size = Object.keys(word2id).length;
    return {
      word2id,
      id2word,
      wids,
      vocab_size,
    };
  }
  static getMergedArray(base:Vector = [], merger:Vector= [], append=false) {
    let arr = new Array().concat(base);
    if (append) arr.splice(base.length-merger.length,merger.length,...merger);
    else arr.splice(0, merger.length, ...merger);
    return arr;
  }
  /**
   */
  static async getContextPairs(this:any, { inputMatrix, vocab_size, window_size = 2, tf, }: { inputMatrix: Matrix; vocab_size: number; window_size: number; tf: any;}) {
    const tensorflow = tf || this.tf;
    const context_length = window_size * 2
    const [emptyXVector, emptyYVector] = await Promise.all([
      tensorflow.zeros([context_length]).array(),
      tensorflow.zeros([vocab_size]).array(),
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
      embedSize: 100,
      windowSize:2,
      ...options
    };
    super(config, properties);
    this.type = 'FeatureEmbedding';
    this.word2id;
    this.id2word;
    this.wids;
    this.vocab_size;
    this.getMergedArray = FeatureEmbedding.getMergedArray;
    this.getFeatureDataSet = FeatureEmbedding.getFeatureDataSet;
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
  generateLayers(this: FeatureEmbedding,x_matrix:Matrix, y_matrix:Matrix, layers?:TensorScriptLayers) {
    const xShape = this.getInputShape(x_matrix);
    const yShape = [this.vocab_size, this.settings.embedSize,];// this.getInputShape(y_matrix);
    this.yShape = yShape;
    this.xShape = xShape;
    const denseLayers:TensorScriptLayers = [];
    if (layers) {
      denseLayers.push(...layers);
    } else {
      /**
       * 
       cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
       */
      denseLayers.push({ inputDim: this.vocab_size, outputDim: this.settings.embedSize, inputLength: this.settings.windowSize * 2, });
      denseLayers.push({ lambdaFunction: 'result = tf.mean(input,1,true)', lambdaOutputShape: [this.vocab_size, this.settings.embedSize] });
      denseLayers.push({ units: this.vocab_size, activation: 'softmax', });
    }
    this.layers = denseLayers;
    console.log({ denseLayers });
    this.tf.serialization.registerClass(lambdaLayer);
    this.model.add(this.tf.layers.embedding(denseLayers[0]));
    // this.model.add(new lambdaLayer(denseLayers[1]));
    this.model.add(this.tf.layers.flatten());
    this.model.add(this.tf.layers.dense(denseLayers[2]));
  }
  async train(x_matrix: Matrix, y_matrix:Matrix, layers?: DenseLayer[]) {
    const featureEmbedDataSet = await this.getFeatureDataSet({ inputMatrixFeatures: x_matrix, });
    this.word2id = featureEmbedDataSet.word2id;
    this.id2word = featureEmbedDataSet.id2word;
    this.wids = featureEmbedDataSet.wids;
    this.vocab_size = featureEmbedDataSet.vocab_size;
    const cxt = await this.getContextPairs({ tf:this.tf, vocab_size: this.vocab_size, inputMatrix: this.wids });
    const x_input_matrix = cxt.x;
    const y_output_matrix = cxt.y;
    // let yShape;
    // let x_matrix;
    // let y_matrix;
    // const look_back = this.settings.lookback;
    // if (y_timeseries) {
    //   x_matrix = x_timeseries;
    //   y_matrix = y_timeseries;
    // } else {
    //   const matrices = this.createDataset(x_timeseries, look_back);
    //   x_matrix = matrices[ 0 ];
    //   y_matrix = matrices[ 1 ];
    //   yShape = this.getInputShape(y_matrix);
    // }
    // //_samples, _timeSteps, _features
    // const timeseriesShape = this.getTimeseriesShape(x_matrix);
    // const x_matrix_timeseries = BaseNeuralNetwork.reshape(x_matrix, timeseriesShape);
    // const xs = this.tf.tensor(x_matrix_timeseries, timeseriesShape);
    // const ys = this.tf.tensor(y_matrix, yShape);
    // this.xShape = timeseriesShape;
    // this.yShape = yShape;
    if (this.compiled === false) {
      this.model = this.tf.sequential();
      //@ts-ignore
      this.generateLayers.call(this, x_input_matrix, y_output_matrix, layers || this.layers,
        // x_test, y_test
      );
      this.model.compile(this.settings.compile);
      this.compiled = true;
    }
    let loss=NaN;
    if(this.settings.fit?.callbacks?.onTrainBegin)this.settings.fit?.callbacks?.onTrainBegin({ loss });
    await asyncForEach(range(1, this.settings.fit?.epochs), async (epoch:number) => {
      if (this.settings.fit?.callbacks?.onEpochBegin) this.settings.fit?.callbacks?.onEpochBegin(epoch, { loss });
      await asyncForEach(x_input_matrix, async (x_input, xIndex) => {
        if (this.settings.fit?.callbacks?.onBatchBegin) this.settings.fit?.callbacks?.onBatchBegin(xIndex, { loss });
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
        // console.log({ x_input, xIndex, y_output, loss })
        xs.dispose();
        ys.dispose();
      });
      if (this.settings.fit?.callbacks?.onEpochEnd) this.settings.fit?.callbacks?.onEpochEnd(epoch, { loss });
    });
    if(this.settings.fit?.callbacks?.onTrainEnd)this.settings.fit?.callbacks?.onTrainEnd({ loss });
    

    // await this.model.fit(xs, ys, this.settings.fit);
    this.trained = true;

    // // this.model.summary();
    // xs.dispose();
    // ys.dispose();
    return this.model;
  }
  // async calculate(x_matrix: Matrix | Vector | InputTextArray) {
  async calculate() {
    return this.model.getWeights()[0];
    console.log('calculate arguments', arguments);
    // console.log('this.model.layers[0].getWeights()',this.model.layers[0].getWeights())
    const weights = this.model.getWeights()[0];
    console.log('in calc',{ weights });
    const arr = await weights.data()
    // const arrM = this.reshape(arr,weights.shape)
    console.log({ arr });
    return weights.data();
    // return weights.toTensor();
    // // const weights = this.model.layers[0].getWeights();
    // return arr;
    // // return super.calculate(input_matrix);
  }
  // async predict(input_matrix: any[], options: PredictionOptions | undefined) {
  async predict(options: PredictionOptions = {}) {
    const predictions = await this.calculate();
    if (options.json === false) {
      return predictions.data();
    } else {
      const arr = await predictions.array()
      if (!this.yShape) throw new Error('Model is missing yShape');
      return this.reshape(arr, predictions.shape);
    }
  }
  labelWeights(weights: Matrix) {
    return weights.reduce((result, weight, index) => { 
      result[this.id2word[index]] = weight;
      return result;
    }, {});
  }
}