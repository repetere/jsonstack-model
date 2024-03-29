import { TensorScriptOptions, TensorScriptProperties, Matrix, Vector, TensorScriptSavedLayers, NestedArray, InputTextArray, PredictionOptions, Shape, TensorScriptLSTMModelContext, } from './model_interface';
import { BaseNeuralNetwork, } from './base_neural_network';
import range from 'lodash.range';

export type TimeSeriesShapeContext = {
  settings: TensorScriptOptions;
  getInputShape: ( ...args: any[] ) => Shape;
}

/**
 * Long Short Term Memory Time Series with Tensorflow
 * @class LSTMTimeSeries
 * @implements {BaseNeuralNetwork}
 */
export class LSTMTimeSeries extends BaseNeuralNetwork {
  declare layers?: TensorScriptSavedLayers;
  // settings: TensorScriptOptions;

  /**
   * Creates dataset data
   * @example
   * LSTMTimeSeries.createDataset([ [ 1, ], [ 2, ], [ 3, ], [ 4, ], [ 5, ], [ 6, ], [ 7, ], [ 8, ], [ 9, ], [ 10, ], ], 3) // => 
      //  [ 
      //    [ 
      //      [ [ 1 ], [ 2 ], [ 3 ] ],
      //      [ [ 2 ], [ 3 ], [ 4 ] ],
      //      [ [ 3 ], [ 4 ], [ 5 ] ],
      //      [ [ 4 ], [ 5 ], [ 6 ] ],
      //      [ [ 5 ], [ 6 ], [ 7 ] ],
      //      [ [ 6 ], [ 7 ], [ 8 ] ], 
      //   ], //x_matrix
      //   [ [ 4 ], [ 5 ], [ 6 ], [ 7 ], [ 8 ], [ 9 ] ] //y_matrix
      // ]
   * @param {Array<Array<number>} dataset - array of values
   * @param {Number} look_back - number of values in each feature 
   * @return {[Array<Array<number>>,Array<number>]} returns x matrix and y matrix for model trainning
   */
  /* istanbul ignore next */
  static createDataset(dataset=[], look_back = 1) { 
    const dataX = new Array();
    const dataY = new Array();
    for (let index in range(dataset.length - look_back - 1)) {
      let i = parseInt(index);
      let a = dataset.slice(i, i + look_back);
      dataX.push(a);
      dataY.push(dataset[ i + look_back ]);
    }
    return [dataX, dataY, ];
  }
  /**
   * Reshape input to be [samples, time steps, features]
   * @example
   * LSTMTimeSeries.getTimeseriesShape([ 
      [ [ 1 ], [ 2 ], [ 3 ] ],
      [ [ 2 ], [ 3 ], [ 4 ] ],
      [ [ 3 ], [ 4 ], [ 5 ] ],
      [ [ 4 ], [ 5 ], [ 6 ] ],
      [ [ 5 ], [ 6 ], [ 7 ] ],
      [ [ 6 ], [ 7 ], [ 8 ] ], 
    ]) //=> [6, 1, 3,]
   * @param {Array<Array<number>} x_timeseries - dataset array of values
   * @return {Array<Array<number>>} returns proper timeseries forecasting shape
   */
  static getTimeseriesShape(this:TimeSeriesShapeContext, x_timeseries:NestedArray<any> | undefined):Shape {
    const time_steps = this.settings.timeSteps;
    const xShape = this.getInputShape(x_timeseries);
    //@ts-ignore
    const _samples = xShape[ 0 ];
    const _timeSteps = time_steps;
    //@ts-ignore
    const _features = xShape[ 1 ];
    const newShape = (this.settings.mulitpleTimeSteps || this.settings.stateful)
      ? [_samples,  _features, _timeSteps, ]
      : [ _samples, _timeSteps, _features, ];
    // console.log({newShape})
    return newShape;
  }
  /**
   * Returns data for predicting values
   * @param timeseries 
   * @param look_back 
   */
  static getTimeseriesDataSet(this:TensorScriptLSTMModelContext, timeseries: never[] | undefined, look_back: any) {
    const lookback = look_back || this.settings.lookback;
    const matrices = LSTMTimeSeries.createDataset.call(this, timeseries, lookback);
    const x_matrix = matrices[ 0 ];
    const y_matrix = matrices[ 1 ];
    // const timeseriesShape = LSTMTimeSeries.getTimeseriesShape.call(this,x_matrix);
    //@ts-ignore
    const x_matrix_timeseries = BaseNeuralNetwork.reshape(x_matrix, [x_matrix.length, lookback, ]);
    const xShape = BaseNeuralNetwork.getInputShape(x_matrix_timeseries);
    const yShape = BaseNeuralNetwork.getInputShape(y_matrix);
    return {
      yShape,
      xShape,
      y_matrix,
      x_matrix:x_matrix_timeseries,
    };
  }
  createDataset: (...args: any[])=>NestedArray<number>;
  getTimeseriesShape: (...args: any[])=>any;
  getTimeseriesDataSet: (...args: any[])=>any;
  /**
   * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options:TensorScriptOptions = {}, properties?:TensorScriptProperties) {
    const config = Object.assign({
      layers: [],
      type: 'simple',
      stateful:false,
      stacked: false,
      mulitpleTimeSteps:false,
      lookback:1,
      batchSize:1,
      timeSteps:1,
      learningRate:0.1,
      compile: {
        loss: 'meanSquaredError',
        optimizer: 'adam',
      },
      fit: {
        epochs: 100,
        batchSize: 1,
      },
    }, options);
    super(config, properties);
    this.type = 'LSTMTimeSeries';

    this.createDataset = LSTMTimeSeries.createDataset;
    this.getTimeseriesDataSet = LSTMTimeSeries.getTimeseriesDataSet;
    this.getTimeseriesShape = LSTMTimeSeries.getTimeseriesShape;
    return this;
  }
  /**
   * Adds dense layers to tensorflow classification model
   * @override 
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @param {Array<Object>} layers - model dense layer parameters
   * @param {Array<Array<number>>} x_test - validation data independent variables
   * @param {Array<Array<number>>} y_test - validation data dependent variables
   */
  generateLayers(this: TensorScriptLSTMModelContext,x_matrix:Matrix, y_matrix:Matrix, layers:TensorScriptSavedLayers) {
    const xShape = this.getInputShape(x_matrix);
    const yShape = this.getInputShape(y_matrix);
    this.yShape = yShape;
    this.xShape = xShape;
    // const sgdoptimizer = this.tf.train.sgd(this.settings.learningRate);
    const lstmLayers = [];
    const rnnLayers = [];
    const denseLayers = [];
    /* istanbul ignore next */
    if (layers) {
      if(layers.lstmLayers && layers.lstmLayers.length) lstmLayers.push(...layers.lstmLayers);
      if(layers.rnnLayers && layers.rnnLayers.length) rnnLayers.push(...layers.rnnLayers);
      if(layers.denseLayers && layers.denseLayers.length) denseLayers.push(...layers.denseLayers);
    } else if (this.settings && this.settings.fit && this.settings.stateful) {
      const batchInputShape = [this.settings.fit.batchSize, this.settings.lookback, 1, ];
      rnnLayers.push({ units: 4, batchInputShape:batchInputShape,  returnSequences:true, });
      rnnLayers.push({ units: 4, batchInputShape:batchInputShape,  });
      denseLayers.push({ units: yShape[1], });
    // } else if(this.settings.stacked) {
    //   lstmLayers.push({ units: 4, inputShape: [ 1, this.settings.lookback ], });
    //   // model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    //   // model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    //   denseLayers.push({ units: yShape[1], });
    } else {
      const inputShape= [1, this.settings.lookback, ];
      // console.log('default timeseries', { inputShape, xShape, yShape ,  });
      lstmLayers.push({ units:4, inputShape,  });
      denseLayers.push({ units: yShape[1], });
    }
    // console.log('lstmLayers',lstmLayers)
    // console.log('denseLayers',denseLayers)
    if (lstmLayers.length) {
      lstmLayers.forEach(layer => {
        this.model.add(this.tf.layers.lstm(layer));
      });
    }
    if (rnnLayers.length) {
      /* istanbul ignore next */
      rnnLayers.forEach(layer => {
        this.model.add(this.tf.layers.simpleRNN(layer));
      });
    }
    if (denseLayers.length) {
      denseLayers.forEach(layer => {
        this.model.add(this.tf.layers.dense(layer));
      });
    }
    this.layers = {
      lstmLayers,
      rnnLayers,
      denseLayers,
    };
    // this.settings.compile.optimizer = sgdoptimizer;
  }
  async train(x_timeseries: any, y_timeseries: any, layers: any, x_test: any, y_test: any) {
    let yShape;
    let x_matrix;
    let y_matrix;
    const look_back = this.settings.lookback;
    if (y_timeseries) {
      x_matrix = x_timeseries;
      y_matrix = y_timeseries;
    } else {
      const matrices = this.createDataset(x_timeseries, look_back);
      x_matrix = matrices[ 0 ];
      y_matrix = matrices[ 1 ];
      yShape = this.getInputShape(y_matrix);
    }
    //_samples, _timeSteps, _features
    const timeseriesShape = this.getTimeseriesShape(x_matrix);
    const x_matrix_timeseries = BaseNeuralNetwork.reshape(x_matrix, timeseriesShape);
    const xs = this.tf.tensor(x_matrix_timeseries, timeseriesShape);
    const ys = this.tf.tensor(y_matrix, yShape);
    this.xShape = timeseriesShape;
    this.yShape = yShape;
    if (this.compiled === false) {
      this.model = this.tf.sequential();
      //@ts-ignore
      this.generateLayers.call(this, x_matrix_timeseries, y_matrix, layers || this.layers,
        // x_test, y_test
      );
      this.model.compile(this.settings.compile);
      if (this.settings.fit && this.settings.stateful) {
        this.settings.fit.shuffle = false;
      }
      this.compiled = true;
    }
    await this.model.fit(xs, ys, this.settings.fit);
    this.trained = true;

    // this.model.summary();
    xs.dispose();
    ys.dispose();
    return this.model;
  }
  calculate(x_matrix: Matrix | Vector | InputTextArray) {
    const timeseriesShape = this.getTimeseriesShape(x_matrix);
    //@ts-ignore
    const input_matrix = BaseNeuralNetwork.reshape(x_matrix, timeseriesShape);
    return super.calculate(input_matrix);
  }
  async predict(input_matrix: Matrix | Vector | InputTextArray, options: PredictionOptions = {}) {
    if (this.settings.stateful && input_matrix.length > 1) {
      //@ts-ignore
      return Promise.all(input_matrix.map((input: string | number | Vector)=>super.predict([input, ], options))) ;
    } else {
      return super.predict.call(this,input_matrix, options);
    }
  }
}