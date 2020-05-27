import * as tf from '@tensorflow/tfjs-node';
import { Tensor, Rank, Shape as TFShape } from '@tensorflow/tfjs-node';
export interface TensorScriptContext {
    type?: string;
    settings: TensorScriptOptions;
    trained: boolean;
    compiled: boolean;
    xShape?: number[];
    yShape?: number[];
    layers?: TensorScriptLayers | TensorScriptSavedLayers;
}
export interface TensorScriptModelContext extends TensorScriptContext {
    model: any;
    tf: any;
    reshape: (...args: any[]) => any;
    getInputShape: (...args: any[]) => any;
}
export interface TensorScriptLSTMModelContext extends TensorScriptModelContext {
    createDataset: (...args: any[]) => any;
    getTimeseriesShape: (...args: any[]) => any;
    layers?: TensorScriptSavedLayers;
}
export declare type TensorScriptProperties = {
    model?: any;
    tf?: any;
};
export declare type DenseLayer = {
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
    lambdaFunction?: string;
    lambdaOutputShape?: Matrix | Vector;
};
export declare type TensorScriptLayers = DenseLayer[];
export declare type TensorScriptSavedLayers = {
    lstmLayers?: DenseLayer[];
    denseLayers?: DenseLayer[];
    rnnLayers?: DenseLayer[];
};
export declare type EpochLog = {
    loss: number;
};
export declare type TensorScriptOptions = {
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
            onTrainBegin?: (logs: EpochLog) => void;
            onTrainEnd?: (logs: EpochLog) => void;
            onEpochBegin?: (epoch: number, logs: EpochLog) => void;
            onEpochEnd?: (epoch: number, logs: EpochLog) => void;
            onBatchBegin?: (batch: number, logs: EpochLog) => void;
            onBatchEnd?: (batch: number, logs: EpochLog) => void;
            onYield?: (epoch: number, batch: number, logs: EpochLog) => void;
        };
    };
    type?: string;
    stateful?: boolean;
    timeSteps?: number;
    mulitpleTimeSteps?: boolean;
    lookback?: number;
    features?: number;
    outputs?: number;
    learningRate?: number;
    PAD?: string;
    embedSize?: number;
    windowSize?: number;
};
export declare type PredictionOptions = {
    skip_matrix_check?: boolean;
    json?: boolean;
    probability?: boolean;
};
export declare type InputTextArray = Array<string>;
export interface NestedArray<T> extends Array<T | NestedArray<T>> {
}
export declare type Shape = Array<number> | number;
export declare type Vector = number[];
export declare type Matrix = Vector[];
export declare type Features = Array<string | number>;
export declare type Corpus = Array<Features>;
export declare type Calculation = {
    data: () => Promise<Vector>;
};
export declare type LambdaLayerOptions = {
    name?: string;
    lambdaFunction: string;
    lambdaOutputShape?: Matrix | Vector;
};
/******************************************************************************
 * tensorflow.js lambda layer
 * written by twitter.com/benjaminwegener
 * license: MIT
 * @see https://benjamin-wegener.blogspot.com/2020/02/tensorflowjs-lambda-layer.html
 */
export declare class LambdaLayer extends tf.layers.Layer {
    name: string;
    lambdaFunction: string;
    lambdaOutputShape: TFShape | Matrix | Vector;
    constructor(config: LambdaLayerOptions);
    call(input: Tensor<Rank> | Tensor<Rank>[], kwargs?: any): Tensor<Rank> | Tensor<Rank>[];
    computeOutputShape(inputShape: Matrix): Matrix | tf.Shape;
    getConfig(): {
        lambdaFunction: string;
        lambdaOutputShape: Matrix | Vector | tf.Shape;
    };
    static get className(): string;
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
export declare class TensorScriptModelInterface {
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
    /**
     * @param {Object} options - tensorflow model hyperparameters
     * @param {Object} customTF - custom, overridale tensorflow / tensorflow-node / tensorflow-node-gpu
     * @param {{model:Object,tf:Object,}} properties - extra instance properties
     */
    constructor(options?: TensorScriptOptions, properties?: TensorScriptProperties);
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
    static reshape(array: Vector, shape: Shape): Matrix | Vector;
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
    static getInputShape(matrix?: any): Shape;
    exportConfiguration(): TensorScriptContext;
    importConfiguration(configuration: TensorScriptContext): void;
    /**
     * Asynchronously trains tensorflow model, must be implemented by tensorscript class
     * @abstract
     * @param {Array<Array<number>>} x_matrix - independent variables
     * @param {Array<Array<number>>} y_matrix - dependent variables
     * @return {Object} returns trained tensorflow model
     */
    train(x_matrix: Matrix, y_matrix?: Matrix, layers?: TensorScriptLayers, x_test?: Matrix, y_test?: Matrix): Promise<tf.LayersModel>;
    train(x_matrix: Matrix, y_matrix: Matrix): Promise<Matrix>;
    /**
     * Predicts new dependent variables
     * @abstract
     * @param {Array<Array<number>>|Array<number>} matrix - new test independent variables
     * @return {{data: Promise}} returns tensorflow prediction
     */
    calculate(matrix: Matrix | Vector | InputTextArray): Calculation;
    /**
     * Loads a saved tensoflow / keras model, this is an alias for
     * @param {Object} options - tensorflow load model options
     * @return {Object} tensorflow model
     * @see {@link https://www.tensorflow.org/js/guide/save_load#loading_a_tfmodel}
     */
    loadModel(options: string): Promise<any>;
    /**
     * saves a tensorflow model, this is an alias for
     * @param {Object} options - tensorflow save model options
     * @return {Object} tensorflow model
     * @see {@link https://www.tensorflow.org/js/guide/save_load#save_a_tfmodel}
     */
    saveModel(options: string): Promise<any>;
    /**
     * Returns prediction values from tensorflow model
     * @param {Array<Array<number>>|Array<number>} input_matrix - new test independent variables
     * @param {Boolean} [options.json=true] - return object instead of typed array
     * @param {Boolean} [options.probability=true] - return real values instead of integers
     * @param {Boolean} [options.skip_matrix_check=false] - validate input is a matrix
     * @return {Array<number>|Array<Array<number>>} predicted model values
     */
    predict(input_matrix?: Matrix | Vector | InputTextArray | PredictionOptions, options?: PredictionOptions): Promise<any>;
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
export declare function size(x: number): number[];
/**
 * Iteratively re-shape a multi dimensional array to fit the specified dimensions (from math.js)
 * @param {Array} array           Array to be reshaped
 * @param {Array.<number>} sizes  List of sizes for each dimension
 * @returns {Array}               Array whose data has been formatted to fit the
 *                                specified dimensions
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 */
export declare function _reshape(array: number[], sizes: number[]): NestedArray<Array<number>>;
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
export declare class DimensionError extends RangeError {
    actual: Shape;
    expected: Shape;
    relation: string;
    isDimensionError: boolean;
    constructor(actual: Shape, expected: Shape, relation: string);
}
/**
 * Flatten a multi dimensional array, put all elements in a one dimensional
 * array
 * @param {Array} array   A multi dimensional array
 * @ignore
 * @see {https://github.com/josdejong/mathjs/blob/develop/src/utils/array.js}
 * @return {Array}        The flattened array (1 dimensional)
 */
export declare function flatten(array: NestedArray<number>): Vector;
export declare function asyncForEach(array: Array<any>, callback: (item: any, index: number, arr: Array<any>) => Promise<any>): Promise<void>;
