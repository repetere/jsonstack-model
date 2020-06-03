// import * as tensorflow from '@tensorflow/tfjs';
// import '@tensorflow/tfjs-node';
// import * as tensorflow from '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs-node';
import { Tensor, Rank, Shape as TFShape } from '@tensorflow/tfjs-node';
// console.log({tensorflow})
/* fix for rollup */
/* istanbul ignore next */
// const tf = (tensorflow && tensorflow.default) ? tensorflow.default : tensorflow;

export interface TensorScriptContext { 
  type?: string;
  settings: TensorScriptOptions;
  trained: boolean;
  compiled: boolean;
  xShape?: number[];
  yShape?: number[];
  layers?: TensorScriptLayers | TensorScriptSavedLayers;
}
export interface TensorScriptModelContext extends TensorScriptContext{
  model: any;
  tf: any;
  reshape: (...args: any[]) => any;
  getInputShape:(...args: any[]) => any;
};

export interface TensorScriptLSTMModelContext extends TensorScriptModelContext{
  createDataset: (...args: any[]) => any;
  getTimeseriesShape: (...args: any[]) => any;
  layers?: TensorScriptSavedLayers;
}

export type TensorScriptProperties = {
  model?: any;
  tf?: any;
};
// export type LambdaLayer = (...args: any[]) => any;
export type DenseLayer = {
  units?: number;
  inputDim?: number;
  outputDim?: number;
  inputLength?: number;
  activation?: string;
  kernelInitializer?: string;
  kernelRegularizer?: any;
  inputShape?: any;
  batchInputShape?: any;
  returnSequences?: boolean;
  // [index: function]
  lambdaFunction?: string;
  lambdaOutputShape?: Matrix|Vector;
};

// tf.layers.add((x)=>tf.mean(x,1))
export type TensorScriptLayers = DenseLayer[];
export type TensorScriptSavedLayers = {
  lstmLayers?: DenseLayer[];
  denseLayers?: DenseLayer[];
  rnnLayers?: DenseLayer[];
};

export type EpochLog = {
  loss: number;
}
export type TensorScriptOptions = {
  name?: string;
  layers?: TensorScriptLayers | TensorScriptSavedLayers;
  layerPreference?: string;
  compile?: {
    loss?: string;
    optimizer?: string;
  };
  fit?: {
    epochs?: number;
    batchSize?: number;
    verbose?: number;
    validationData?: [Matrix, Matrix];
    validation_data?: [Matrix, Matrix];
    shuffle?: boolean;
    callbacks?: {
      // called when training begins.
      onTrainBegin?: (logs:EpochLog) => void;
      // called when training ends.
      onTrainEnd?: (logs:EpochLog) => void;
      //called at the start of every epoch.
      onEpochBegin?: (epoch: number, logs: EpochLog) => void;
      //called at the end of every epoch.
      onEpochEnd?: (epoch: number, logs: EpochLog) => void;
      //called at the start of every batch.
      onBatchBegin?:(batch:number, logs: EpochLog)=> void;
      //called at the end of every batch.
      onBatchEnd?:(batch:number, logs: EpochLog)=> void;
      //called every yieldEvery milliseconds with the current epoch, batch and logs. The logs are the same as in onBatchEnd(). Note that onYield can skip batches or epochs. See also docs for yieldEvery below.
      onYield?:(epoch:number, batch:number, logs: EpochLog)=> void;
    }
  }
  //logistic regression
  type?: string;
  //LSTM Options
  stateful?: boolean;
  timeSteps?: number;
  mulitpleTimeSteps?: boolean;
  lookback?: number;
  features?: number;
  outputs?: number;
  learningRate?: number;
  //Embedding options
  PAD?: string;
  embedSize?: number;
  windowSize?: number;
  streamInputMatrix?: boolean;
};

export type PredictionOptions = {
  skip_matrix_check?: boolean;
  json?: boolean;
  probability?: boolean;
};

export type InputTextArray = Array<string>;
export interface NestedArray<T> extends Array<T | NestedArray<T>> { }
export type Shape = Array<number>|number;
export type Vector = number[];
export type Matrix = Vector[];

export type Features = Array<string | number>
export type Corpus = Array<Features>
// export type DataCalculation = ()=>Promise<Vector>
// export type DataCalculation = {
//   data:()=>Promise<Vector>
// }
export type Calculation = {
  data: ()=>Promise<Vector>;
}

export type LambdaLayerOptions = {
  name?: string;
  lambdaFunction: string;
  lambdaOutputShape?: Matrix|Vector;
}
/******************************************************************************
 * tensorflow.js lambda layer
 * written by twitter.com/benjaminwegener
 * license: MIT
 * @see https://benjamin-wegener.blogspot.com/2020/02/tensorflowjs-lambda-layer.html
 */
export class LambdaLayer extends tf.layers.Layer {
  name: string;
  lambdaFunction: string;
  //@ts-ignore
  lambdaOutputShape: TFShape | Matrix | Vector;
  constructor(config:LambdaLayerOptions) {
    super(config);
    if (config.name === undefined) {
        config.name = ((+new Date) * Math.random()).toString(36); //random name from timestamp in case name hasn't been set
    }
    this.name = config.name;
    this.lambdaFunction = config.lambdaFunction;
    if(config.lambdaOutputShape) this.lambdaOutputShape = config.lambdaOutputShape;
  }

  call(input: Tensor<Rank> | Tensor<Rank>[], kwargs?: any): Tensor<Rank> | Tensor<Rank>[] {
    // console.log({ input }, 'input[0].shape', input[0].shape)
    // input[0].data().then(inputData=>console.log)
    // console.log('input[0].data()', input[0].data())
    // return input;
    return tf.tidy(() => {
      // return tf.mean(tf.tensor(input),1,true)
      let result = new Array();
      // eval(this.lambdaFunction);
      result = (new Function('input', 'tf', this.lambdaFunction))(input, tf);
      // result = tf.mean(input,1);
      return result;
    });
  }

  computeOutputShape(inputShape:Matrix) {
    // console.log('computeOutputShape',{inputShape})
    if (this.lambdaOutputShape === undefined) { //if no outputshape provided, try to set as inputshape
      return inputShape[0];
    } else {
      return this.lambdaOutputShape;
    }
  }

  getConfig() {
    const config = {
      ...super.getConfig(),
      lambdaFunction: this.lambdaFunction,
      lambdaOutputShape: this.lambdaOutputShape,
    };
    return config;
  }

  static get className():string {
      return 'LambdaLayer';
  }
}

/**
 * Base class for tensorscript models
 * @interface TensorScriptModelInterface
 * @property {Object} settings - tensorflow model hyperparameters
 * @property {Object} model - tensorflow model
 * @property {Object} tf - tensorflow / tensorflow-node / tensorflow-node-gpu
 * @property {Function} reshape - static reshape array function
 * @property {Function} getInputShape - static TensorScriptModelInterface
 */
export class TensorScriptModelInterface  {
  type: string;
  settings: TensorScriptOptions;
  model: any;
  tokenizer: any;
  tf: any;
  trained: boolean;
  compiled: boolean;
  reshape: (...args: any[]) => any;
  getInputShape: (...args: any[]) => any;
  xShape?: number[];
  yShape?: number[];
  layers?: TensorScriptLayers | TensorScriptSavedLayers;
  getTimeseriesShape?: (x_timeseries: NestedArray<any> | undefined) => Shape;
  loss?: number;
  /**
   * @param {Object} options - tensorflow model hyperparameters
   * @param {Object} customTF - custom, overridale tensorflow / tensorflow-node / tensorflow-node-gpu
   * @param {{model:Object,tf:Object,}} properties - extra instance properties
   */
  constructor(options:TensorScriptOptions = {}, properties:TensorScriptProperties = {}) {
    // tf.setBackend('cpu');
    this.type = 'ModelInterface';
    /** @type {Object} */
    this.settings = Object.assign({  }, options);
    /** @type {Object} */
    this.model = properties.model;
    /** @type {Object} */
    this.tf = properties.tf || tf;
    /** @type {Boolean} */
    this.trained = false;
    this.compiled = false;
    /** @type {Function} */
    this.reshape = TensorScriptModelInterface.reshape;
    /** @type {Function} */
    this.getInputShape = TensorScriptModelInterface.getInputShape;
    if( this.tf && this.tf.serialization && this.tf.serialization.registerClass) this.tf.serialization.registerClass(LambdaLayer);
    return this;
  }
  /**
   * Reshapes an array
   * @function
   * @example 
   * const array = [ 0, 1, 1, 0, ];
   * const shape = [2,2];
   * TensorScriptModelInterface.reshape(array,shape) // => 
   * [
   *   [ 0, 1, ],
   *   [ 1, 0, ],
   * ];
   * @param {Array<number>} array - input array 
   * @param {Array<number>} shape - shape array 
   * @return {Array<Array<number>>} returns a matrix with the defined shape
   */
  /* istanbul ignore next */
  static reshape(array:Vector, shape:Shape):Matrix|Vector {
    const flatArray:number[] = flatten(array);
   

    function product (arr:Vector) {
      return arr.reduce((prev, curr) => prev * curr);
    }
  
    if (!Array.isArray(array) || !Array.isArray(shape)) {
      throw new TypeError('Array expected');
    }
  
    if (shape.length === 0) {
      //@ts-ignore
      throw new DimensionError(0, product(size(array)), '!=');
    }
    let newArray;
    let totalSize = 1;
    const rows = shape[ 0 ];
    for (let sizeIndex = 0; sizeIndex < shape.length; sizeIndex++) {
      totalSize *= shape[sizeIndex];
    }
  
    if (flatArray.length !== totalSize) {
      throw new DimensionError(
        product(shape),
        //@ts-ignore
        product(size(array)),
        '!='
      );
    }
  
    try {
      newArray = _reshape(flatArray, shape);
    } catch (e) {
      if (e instanceof DimensionError) {
        throw new DimensionError(
          product(shape),
          //@ts-ignore
          product(size(array)),
          '!='
        );
      }
      throw e;
    }
    if (newArray.length !== rows) throw new SyntaxError(`specified shape (${shape}) is compatible with input array or length (${array.length})`);

    // console.log({ newArray ,});
    //@ts-ignore
    return newArray;
  }
  /**
   * Returns the shape of an input matrix
   * @function
   * @example 
   * const input = [
   *   [ 0, 1, ],
   *   [ 1, 0, ],
   * ];
   * TensorScriptModelInterface.getInputShape(input) // => [2,2]
   * @see {https://stackoverflow.com/questions/10237615/get-size-of-dimensions-in-array}
   * @param {Array<Array<number>>} matrix - input matrix 
   * @return {Array<number>} returns the shape of a matrix (e.g. [2,2])
   */
  //@ts-ignore
  static getInputShape(matrix:any=[]):Shape {
  // static getInputShape(matrix:NestedArray<V>=[]):Shape {
    if (Array.isArray(matrix) === false || !matrix[ 0 ] || !matrix[ 0 ].length || Array.isArray(matrix[ 0 ]) === false) throw new TypeError('input must be a matrix');
    const dim = [];
    const x_dimensions = matrix[ 0 ].length;
    let vectors:any = matrix;
    matrix.forEach((vector:Vector) => {
      if (vector.length !== x_dimensions) throw new SyntaxError('input must have the same length in each row');
    });
    for (;;) {
      dim.push(vectors.length);
      if (Array.isArray(vectors[0])) {
        vectors = vectors[0];
      } else {
        break;
      }
    }
    return dim;
  }
  exportConfiguration():TensorScriptContext {
    return {
      type: this.type,
      settings: this.settings,
      trained: this.trained,
      compiled: this.compiled,
      xShape: this.xShape,
      yShape: this.yShape,
      layers: this.layers,
    };
  }
  importConfiguration(configuration:TensorScriptContext):void {
    this.type = configuration.type || this.type;
    this.settings = {
      ...this.settings,
      ...configuration.settings,
    };
    this.trained = configuration.trained || this.trained;
    this.compiled = configuration.compiled || this.compiled;
    this.xShape = configuration.xShape || this.xShape;
    this.yShape = configuration.yShape || this.yShape;
    this.layers = configuration.layers || this.layers;
  }
  /**
   * Asynchronously trains tensorflow model, must be implemented by tensorscript class
   * @abstract 
   * @param {Array<Array<number>>} x_matrix - independent variables
   * @param {Array<Array<number>>} y_matrix - dependent variables
   * @return {Object} returns trained tensorflow model 
   */
  async train(x_matrix: Matrix, y_matrix?: Matrix, layers?: TensorScriptLayers, x_test?: Matrix, y_test?: Matrix): Promise<tf.LayersModel>
  async train(x_matrix:Matrix, y_matrix:Matrix):Promise<Matrix>
  train(x_matrix:Matrix, y_matrix:Matrix):any {
    throw new ReferenceError('train method is not implemented');
  }
  /**
   * Predicts new dependent variables
   * @abstract 
   * @param {Array<Array<number>>|Array<number>} matrix - new test independent variables
   * @return {{data: Promise}} returns tensorflow prediction 
   */
  calculate(matrix:Matrix|Vector|InputTextArray): Calculation {
    throw new ReferenceError('calculate method is not implemented');
  }
  /**
   * Loads a saved tensoflow / keras model, this is an alias for 
   * @param {Object} options - tensorflow load model options
   * @return {Object} tensorflow model
   * @see {@link https://www.tensorflow.org/js/guide/save_load#loading_a_tfmodel}
   */
  async loadModel(options: string) {
    this.model = await this.tf.loadLayersModel(options);
    this.xShape = this.model.inputs[0].shape;
    this.yShape = this.model.outputs[0].shape;
    this.trained = true;
    return this.model;
  }
  /**
   * saves a tensorflow model, this is an alias for 
   * @param {Object} options - tensorflow save model options
   * @return {Object} tensorflow model
   * @see {@link https://www.tensorflow.org/js/guide/save_load#save_a_tfmodel}
   */
  async saveModel(options:string) {
    const savedStatus = await this.model.save(options);
    return savedStatus;
  }

  /**
   * Returns prediction values from tensorflow model
   * @param {Array<Array<number>>|Array<number>} input_matrix - new test independent variables 
   * @param {Boolean} [options.json=true] - return object instead of typed array
   * @param {Boolean} [options.probability=true] - return real values instead of integers
   * @param {Boolean} [options.skip_matrix_check=false] - validate input is a matrix
   * @return {Array<number>|Array<Array<number>>} predicted model values
   */
  // async predict(options?:Matrix|Vector|InputTextArray|PredictionOptions):Promise<Matrix> 
  async predict(input_matrix?: Matrix | Vector | InputTextArray | PredictionOptions, options?: PredictionOptions) {
    if (!input_matrix || Array.isArray(input_matrix) === false) throw new Error('invalid input matrix');
    const config:PredictionOptions = {
      json: true,
      probability: true,
      ...options
    };
    const x_matrix = (Array.isArray(input_matrix as Matrix | Vector[0]) || config.skip_matrix_check)
          ? input_matrix
      : [
        input_matrix,
      ];
    return this.calculate(x_matrix as Matrix)
      .data()
      .then((predictions:Vector) => {
        // console.log({ predictions });
        if (config.json === false) {
          return predictions;
        } else {
          if (!this.yShape) throw new Error('Model is missing yShape');
          const shape = [(x_matrix as Matrix | Vector| InputTextArray).length, this.yShape[ 1 ], ];
          const predictionValues = (config.probability === false) ? Array.from(predictions).map(Math.round) : Array.from(predictions);
          return this.reshape(predictionValues, shape);
        }
      })
      .catch((e:Error) => {
        throw e; 
      });
  }
}

/**
 * Calculate the size of a multi dimensional array.
 * This function checks the size of the first entry, it does not validate
 * whether all dimensions match. (use function `validate` for that) (from math.js)
 * @param {Array} x
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 * @ignore
 * @return {Number[]} size
 */
/* istanbul ignore next */
export function size (x:number):number[] {
  let s = [];

  while (Array.isArray(x)) {
    s.push(x.length);
    x = x[0];
  }

  return s;
}
/**
 * Iteratively re-shape a multi dimensional array to fit the specified dimensions (from math.js)
 * @param {Array} array           Array to be reshaped
 * @param {Array.<number>} sizes  List of sizes for each dimension
 * @returns {Array}               Array whose data has been formatted to fit the
 *                                specified dimensions
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 */
/* istanbul ignore next */
export function _reshape(array:number[], sizes:number[]):NestedArray<Array<number>> {
  // testing if there are enough elements for the requested shape
  var tmpArray = array;
  var tmpArray2:number[][];
  // for each dimensions starting by the last one and ignoring the first one
  for (var sizeIndex = sizes.length - 1; sizeIndex > 0; sizeIndex--) {
    var size = sizes[sizeIndex];
    tmpArray2 = [];

    // aggregate the elements of the current tmpArray in elements of the requested size
    var length = tmpArray.length / size;
    for (var i = 0; i < length; i++) {
      tmpArray2.push(tmpArray.slice(i * size, (i + 1) * size));
    }
    // set it as the new tmpArray for the next loop turn or for return
    //@ts-ignore
    tmpArray = tmpArray2;
  }
  //@ts-ignore
  return tmpArray;
}

/**
 * Create a range error with the message:
 *     'Dimension mismatch (<actual size> != <expected size>)' (from math.js)
 * @param {number | number[]} actual        The actual size
 * @param {number | number[]} expected      The expected size
 * @param {string} [relation='!=']          Optional relation between actual
 *                                          and expected size: '!=', '<', etc.
 * @extends RangeError
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 */
/* istanbul ignore next */
export class DimensionError extends RangeError {
  actual: Shape;
  expected: Shape;
  relation: string;
  isDimensionError: boolean;
  constructor(actual:Shape, expected:Shape, relation:string) {
    /* istanbul ignore next */
    const message = 'Dimension mismatch (' + (Array.isArray(actual) ? ('[' + actual.join(', ') + ']') : actual) + ' ' + ('!=') + ' ' + (Array.isArray(expected) ? ('[' + expected.join(', ') + ']') : expected) +  ')';
    super(message);
  
    this.actual = actual;
    this.expected = expected;
    this.relation = relation;
    // this.stack = (new Error()).stack
    this.message = message;
    this.name = 'DimensionError';
    this.isDimensionError = true;
  }
}

/**
 * Flatten a multi dimensional array, put all elements in a one dimensional
 * array
 * @param {Array} array   A multi dimensional array
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 * @return {Array}        The flattened array (1 dimensional)
 */
/* istanbul ignore next */
export function flatten(array:NestedArray<number>):Vector {
  /* istanbul ignore next */
  if (!Array.isArray(array)) {
    // if not an array, return as is
    /* istanbul ignore next */
    return array;
  }
  let flat:Vector = [];
  
  /* istanbul ignore next */
  array.forEach(function callback(value:NestedArray<number>|number) {
    if (Array.isArray(value)) {
      value.forEach(callback); // traverse through sub-arrays recursively
    } else {
      flat.push(value);
    }
  });

  return flat;
}


export async function asyncForEach(array:Array<any>, callback: (item:any, index:number,arr:Array<any>) => Promise<any>) {
  for (let index = 0; index < array.length; index++) {
    await callback(array[index], index, array);
  }
}