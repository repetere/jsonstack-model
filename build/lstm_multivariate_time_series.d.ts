import { TensorScriptOptions, TensorScriptProperties, Matrix, Vector, TensorScriptLayers, TensorScriptSavedLayers, NestedArray, TensorScriptModelContext, TensorScriptLSTMModelContext } from './model_interface';
import { LSTMTimeSeries } from './lstm_time_series';
/**
 * Long Short Term Memory Multivariate Time Series with Tensorflow
 * @class LSTMMultivariateTimeSeries
 * @extends {LSTMTimeSeries}
 */
export declare class LSTMMultivariateTimeSeries extends LSTMTimeSeries {
    /**
     * Creates dataset data
     * @example
     * const ds = [
    [10, 20, 30, 40, 50, 60, 70, 80, 90,],
    [11, 21, 31, 41, 51, 61, 71, 81, 91,],
    [12, 22, 32, 42, 52, 62, 72, 82, 92,],
    [13, 23, 33, 43, 53, 63, 73, 83, 93,],
    [14, 24, 34, 44, 54, 64, 74, 84, 94,],
    [15, 25, 35, 45, 55, 65, 75, 85, 95,],
    [16, 26, 36, 46, 56, 66, 76, 86, 96,],
    [17, 27, 37, 47, 57, 67, 77, 87, 97,],
    [18, 28, 38, 48, 58, 68, 78, 88, 98,],
    [19, 29, 39, 49, 59, 69, 79, 89, 99,],
  ];
     * LSTMMultivariateTimeSeries.createDataset(ds,1) // =>
        //  [
        //   [
        //    [ 20, 30, 40, 50, 60, 70, 80, 90 ],
        //    [ 21, 31, 41, 51, 61, 71, 81, 91 ],
        //    [ 22, 32, 42, 52, 62, 72, 82, 92 ],
        //    [ 23, 33, 43, 53, 63, 73, 83, 93 ],
        //    [ 24, 34, 44, 54, 64, 74, 84, 94 ],
        //    [ 25, 35, 45, 55, 65, 75, 85, 95 ],
        //    [ 26, 36, 46, 56, 66, 76, 86, 96 ],
        //    [ 27, 37, 47, 57, 67, 77, 87, 97 ],
        //    [ 28, 38, 48, 58, 68, 78, 88, 98 ]
        //   ], //x_matrix
        //   [ 11, 12, 13, 14, 15, 16, 17, 18, 19 ], //y_matrix
        //   8 //features
        // ]
     * @param {Array<Array<number>} dataset - array of values
     * @param {Number} look_back - number of values in each feature
     * @override
     * @return {[Array<Array<number>>,Array<number>]} returns x matrix and y matrix for model trainning
     */
    static createDataset(this: TensorScriptModelContext, dataset?: NestedArray<Array<number>>, look_back?: number): NestedArray<number>;
    /**
     * Drops columns by array index
     * @example
  const data = [ [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 11, 21, 31, 41, 51, 61, 71, 81, 91 ],
       [ 11, 21, 31, 41, 51, 61, 71, 81, 91, 12, 22, 32, 42, 52, 62, 72, 82, 92 ],
       [ 12, 22, 32, 42, 52, 62, 72, 82, 92, 13, 23, 33, 43, 53, 63, 73, 83, 93 ],
       [ 13, 23, 33, 43, 53, 63, 73, 83, 93, 14, 24, 34, 44, 54, 64, 74, 84, 94 ],
       [ 14, 24, 34, 44, 54, 64, 74, 84, 94, 15, 25, 35, 45, 55, 65, 75, 85, 95 ],
       [ 15, 25, 35, 45, 55, 65, 75, 85, 95, 16, 26, 36, 46, 56, 66, 76, 86, 96 ],
       [ 16, 26, 36, 46, 56, 66, 76, 86, 96, 17, 27, 37, 47, 57, 67, 77, 87, 97 ],
       [ 17, 27, 37, 47, 57, 67, 77, 87, 97, 18, 28, 38, 48, 58, 68, 78, 88, 98 ],
       [ 18, 28, 38, 48, 58, 68, 78, 88, 98, 19, 29, 39, 49, 59, 69, 79, 89, 99 ] ];
  const n_in = 1; //lookbacks
  const n_out = 1;
  const dropColumns = getDropableColumns(8, n_in, n_out); // =>[ 10, 11, 12, 13, 14, 15, 16, 17 ]
  const newdata = drop(data,dropColumns); //=> [
      // [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 11 ],
      // [ 11, 21, 31, 41, 51, 61, 71, 81, 91, 12 ],
      // [ 12, 22, 32, 42, 52, 62, 72, 82, 92, 13 ],
      // [ 13, 23, 33, 43, 53, 63, 73, 83, 93, 14 ],
      // [ 14, 24, 34, 44, 54, 64, 74, 84, 94, 15 ],
      // [ 15, 25, 35, 45, 55, 65, 75, 85, 95, 16 ],
      // [ 16, 26, 36, 46, 56, 66, 76, 86, 96, 17 ],
      // [ 17, 27, 37, 47, 57, 67, 77, 87, 97, 18 ],
      // [ 18, 28, 38, 48, 58, 68, 78, 88, 98, 19 ]
      //]
    * @param {Array<Array<number>>} data - data set to drop columns
    * @param {Array<number>} columns - array of column indexes
     * @returns {Array<Array<number>>} matrix with dropped columns
     */
    static drop(data: NestedArray<Array<number>>, columns: Array<number>): NestedArray<Array<number>> | number[];
    /**
     * Converts data set to supervised labels for forecasting, the first column must be the dependent variable
     * @example
     const ds = [
      [10, 20, 30, 40, 50, 60, 70, 80, 90,],
      [11, 21, 31, 41, 51, 61, 71, 81, 91,],
      [12, 22, 32, 42, 52, 62, 72, 82, 92,],
      [13, 23, 33, 43, 53, 63, 73, 83, 93,],
      [14, 24, 34, 44, 54, 64, 74, 84, 94,],
      [15, 25, 35, 45, 55, 65, 75, 85, 95,],
      [16, 26, 36, 46, 56, 66, 76, 86, 96,],
      [17, 27, 37, 47, 57, 67, 77, 87, 97,],
      [18, 28, 38, 48, 58, 68, 78, 88, 98,],
      [19, 29, 39, 49, 59, 69, 79, 89, 99,],
    ];
    const n_in = 1; //lookbacks
    const n_out = 1;
    const series = seriesToSupervised(ds, n_in, n_out); //=> [
      // [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 11, 21, 31, 41, 51, 61, 71, 81, 91 ],
      // [ 11, 21, 31, 41, 51, 61, 71, 81, 91, 12, 22, 32, 42, 52, 62, 72, 82, 92 ],
      // [ 12, 22, 32, 42, 52, 62, 72, 82, 92, 13, 23, 33, 43, 53, 63, 73, 83, 93 ],
      // [ 13, 23, 33, 43, 53, 63, 73, 83, 93, 14, 24, 34, 44, 54, 64, 74, 84, 94 ],
      // [ 14, 24, 34, 44, 54, 64, 74, 84, 94, 15, 25, 35, 45, 55, 65, 75, 85, 95 ],
      // [ 15, 25, 35, 45, 55, 65, 75, 85, 95, 16, 26, 36, 46, 56, 66, 76, 86, 96 ],
      // [ 16, 26, 36, 46, 56, 66, 76, 86, 96, 17, 27, 37, 47, 57, 67, 77, 87, 97 ],
      // [ 17, 27, 37, 47, 57, 67, 77, 87, 97, 18, 28, 38, 48, 58, 68, 78, 88, 98 ],
      // [ 18, 28, 38, 48, 58, 68, 78, 88, 98, 19, 29, 39, 49, 59, 69, 79, 89, 99 ]
      //];
     *
     * @param {Array<Array<number>>} data - data set
     * @param {number} n_in - look backs
     * @param {number} n_out - future iterations (only 1 supported)
     * @todo support multiple future iterations
     * @returns {Array<Array<number>>} multivariate dataset for time series
     */
    static seriesToSupervised(data: number[], n_in?: number, n_out?: number): number[];
    /**
     * Calculates which columns to drop by index
     * @todo support multiple iterations in the future, also only one output variable supported in column features * lookbacks -1
     * @example
  const ds = [
    [10, 20, 30, 40, 50, 60, 70, 80, 90,],
    [11, 21, 31, 41, 51, 61, 71, 81, 91,],
    [12, 22, 32, 42, 52, 62, 72, 82, 92,],
    [13, 23, 33, 43, 53, 63, 73, 83, 93,],
    [14, 24, 34, 44, 54, 64, 74, 84, 94,],
    [15, 25, 35, 45, 55, 65, 75, 85, 95,],
    [16, 26, 36, 46, 56, 66, 76, 86, 96,],
    [17, 27, 37, 47, 57, 67, 77, 87, 97,],
    [18, 28, 38, 48, 58, 68, 78, 88, 98,],
    [19, 29, 39, 49, 59, 69, 79, 89, 99,],
  ];
  const n_in = 1; //lookbacks
  const n_out = 1;
  const dropped = getDropableColumns(8, n_in, n_out); //=> [ 10, 11, 12, 13, 14, 15, 16, 17 ]
     * @param {number} features - number of independent variables
     * @param {number} n_in - look backs
     * @param {number} n_out - future iterations (currently only 1 supported)
     * @returns {Array<number>} array indexes to drop
     */
    static getDropableColumns(features: number, n_in: number, n_out: number): number[];
    /**
     * Reshape input to be [samples, time steps, features]
     * @example
     * @override
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
    static getTimeseriesShape(this: TensorScriptModelContext, x_timeseries: NestedArray<any> | undefined): any[];
    /**
     * Returns data for predicting values
     * @param timeseries
     * @param look_back
     * @override
     */
    static getTimeseriesDataSet(this: TensorScriptModelContext, timeseries: NestedArray<number[]> | undefined, look_back: any): {
        yShape: import("./model_interface").Shape;
        xShape: import("./model_interface").Shape;
        y_matrix: Vector;
        x_matrix: Matrix | Vector;
    };
    createDataset: (...args: any[]) => NestedArray<number>;
    seriesToSupervised: (...args: any[]) => Array<number>;
    getDropableColumns: (...args: any[]) => any;
    drop: (...args: any[]) => any;
    getTimeseriesShape: (...args: any[]) => any;
    getTimeseriesDataSet: (...args: any[]) => any;
    /**
     * @param {{layers:Array<Object>,compile:Object,fit:Object}} options - neural network configuration and tensorflow model hyperparameters
     * @param {{model:Object,tf:Object,}} properties - extra instance properties
     */
    constructor(options?: TensorScriptOptions, properties?: TensorScriptProperties);
    /**
     * Adds dense layers to tensorflow classification model
     * @override
     * @param {Array<Array<number>>} x_matrix - independent variables
     * @param {Array<Array<number>>} y_matrix - dependent variables
     * @param {Array<Object>} layers - model dense layer parameters
     * @param {Array<Array<number>>} x_test - validation data independent variables
     * @param {Array<Array<number>>} y_test - validation data dependent variables
     */
    generateLayers(x_matrix: Matrix, y_matrix: Matrix, layers: TensorScriptSavedLayers): void;
    /**
     * @override
     * @param x_timeseries
     * @param y_timeseries
     * @param layers
     * @param x_test
     * @param y_test
     */
    train(this: TensorScriptLSTMModelContext, x_timeseries: any, y_timeseries: any, layers: TensorScriptLayers, x_test: Matrix, y_test: Matrix): Promise<any>;
}
/**
 * Returns an array of vectors as an array of arrays (from modelscript)
 * @example
const vectors = [ [1,2,3], [1,2,3], [3,3,4], [3,3,3] ];
const arrays = pivotVector(vectors); // => [ [1,2,3,3], [2,2,3,3], [3,3,4,3] ];
 * @memberOf util
 * @param {Array[]} vectors
 * @returns {Array[]}
 * @ignore
 * @see {https://github.com/repetere/modelscript/blob/master/src/util.js}
 */
export declare function pivotVector(vectors?: Matrix): Matrix;
